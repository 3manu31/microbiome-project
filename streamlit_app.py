"""
Streamlit Microbiome Top Microbes Dashboard

This app lets users select a grouping column (healthy, mental illness, sex, sample type) and visualizes the top microbes per group interactively.
"""



import streamlit as st
import pandas as pd
from biom import load_table
from biom.table import Table
import matplotlib.pyplot as plt
import os
import tempfile
import os

# Detect Streamlit Cloud environment
is_cloud = os.environ.get("STREAMLIT_SERVER_HEADLESS", "false").lower() == "true"


st.title("Microbiome Top Microbes Dashboard")

# --- Limitations & Warnings ---
st.sidebar.header("Limitations & Warnings")
st.sidebar.warning("\n- The live demo may be slow or crash if toggling options too quickly due to Streamlit Cloud resource limits.\n- Please toggle one option at a time and wait for the page to load before toggling again.\n- File upload is disabled on the cloud demo; to use this feature, install and run the app locally.\n- If you see errors or the app crashes, reload the page and try again.\n")


# --- File uploaders (only enabled for local runs) ---
if not is_cloud:
    st.sidebar.header("Upload Your Files")
    uploaded_metadata = st.sidebar.file_uploader("Upload metadata file (.txt or .csv)", type=["txt", "csv"])
    uploaded_biom = st.sidebar.file_uploader("Upload biom file (.biom)", type=["biom"])
else:
    uploaded_metadata = None
    uploaded_biom = None


# --- Load metadata with caching ---
@st.cache_data(show_spinner=False)
def load_metadata(uploaded_metadata):
    if uploaded_metadata is not None:
        if hasattr(uploaded_metadata, 'size') and uploaded_metadata.size > 100 * 1024 * 1024:
            st.error("Uploaded metadata file is too large. Please upload a file smaller than 100MB.")
            st.stop()
        return pd.read_csv(
            uploaded_metadata,
            sep='\t' if uploaded_metadata.name.endswith('.txt') else ',',
            low_memory=False,
            encoding='utf-8'
        )
    else:
        # Load distilled demo metadata file
        if not os.path.exists('metadata_demo.txt'):
            st.error("Demo metadata_demo.txt file not found in repo. Please upload a metadata file.")
            st.stop()
        return pd.read_csv('metadata_demo.txt', sep='\t', low_memory=False, encoding='utf-8')

try:
    metadata = load_metadata(uploaded_metadata)
except Exception as e:
    st.error(f"Error loading metadata: {e}")
    st.stop()

# --- Load abundance data from .biom file ---



# --- Load BIOM file and cache DataFrame ---
def parse_biom(_uploaded_biom):
    content = _uploaded_biom.read()
    try:
        import json
        table = Table.from_json(json.loads(content.decode('utf-8')))
    except Exception:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            table = load_table(tmp.name)
    return table.to_dataframe(dense=True).T  # Samples as rows

@st.cache_data(show_spinner=False)
def load_abundance_df(_uploaded_biom):
    return parse_biom(_uploaded_biom)



# --- Load BIOM file for demo mode (cloud) ---
try:
    if uploaded_biom is not None:
        if hasattr(uploaded_biom, 'size') and uploaded_biom.size > 100 * 1024 * 1024:
            st.error("Uploaded BIOM file is too large. Please upload a file smaller than 100MB.")
            st.stop()
        abundance_df = load_abundance_df(uploaded_biom)
    else:
        # Demo mode: load demo_biom.biom file
        if not os.path.exists('demo_biom.biom'):
            st.error("Demo BIOM file not found in repo. Please upload a BIOM file named demo_biom.biom.")
            st.stop()
        abundance_df = parse_biom(open('demo_biom.biom', 'rb'))
        # Sample for resource efficiency
        abundance_df = abundance_df.iloc[:100, :100]
        st.info(f"Loaded demo BIOM file with shape: {abundance_df.shape}")
except Exception as e:
    st.error(f"Error loading biom file: {e}")
    st.stop()

# --- Merge abundance and metadata ---


# Only limit samples/features for demo data
is_demo = (
    uploaded_metadata is None and uploaded_biom is None
)
MAX_FEATURES = 100 if is_demo else None

if metadata is None or abundance_df is None:
    st.warning("Please upload both a metadata file and a BIOM file to proceed.")
    st.stop()
try:
    # Limit features for demo data only
    if is_demo:
        if MAX_FEATURES and abundance_df.shape[1] > MAX_FEATURES:
            abundance_df = abundance_df.iloc[:, :MAX_FEATURES]
    merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id')
except Exception as e:
    st.error(f"Error merging abundance and metadata: {e}. Please check that sample IDs match.")
    st.stop()


# --- Precompute and cache group means for grouped bar chart ---
group_options = [
    ('age_cat', 'Age Category'),
    ('mental_illness', 'Mental Illness'),
    ('sex', 'Sex'),
    ('sample_type', 'Sample Type'),
    ('asd', 'Autism Spectrum Disorder (ASD)')
]


import itertools

@st.cache_data(show_spinner=False)
def precompute_group_combo_means(merged, group_options, metadata):
    cache = {}
    for group_col, group_label in group_options:
        group_values = merged[group_col].dropna().unique().tolist()
        combos = []
        # All non-empty combinations
        for r in range(1, len(group_values)+1):
            combos.extend(itertools.combinations(group_values, r))
        cache[group_col] = {}
        for combo in combos:
            # Combo is a tuple of group values
            combo_df = merged[merged[group_col].isin(combo)]
            mean_abundance = combo_df.iloc[:, :-len(metadata.columns)].mean(axis=0)
            cache[group_col][frozenset(combo)] = mean_abundance
        # Overall mean (all groups)
        overall_mean = merged.iloc[:, :-len(metadata.columns)].mean(axis=0)
        cache[group_col]['All'] = overall_mean
    return cache

cached_group_combo_means = precompute_group_combo_means(merged, group_options, metadata)

# --- UI for grouped bar chart ---

group_col_label = st.selectbox("Select grouping column:", [label for _, label in group_options])
group_col = next(code for code, label in group_options if label == group_col_label)
group_label = group_col_label
group_values = [g for g in merged[group_col].dropna().unique().tolist()]
default_selected = group_values.copy()
selected_groups = st.multiselect(
    f"Show {group_label} options in grouped bar chart:",
    options=group_values,
    default=default_selected,
    help="Toggle which groups to display in the grouped bar chart."
)
top_n = st.slider("Select number of top microbes:", min_value=5, max_value=15, value=10, step=1)


# --- Compute top microbes and prepare comparison table from cached means ---

def get_top_microbes_from_combo_cache(cached_combo_means, selected_groups, top_n):
    # Use frozenset for lookup
    combo_key = frozenset(selected_groups)
    mean_abundance = cached_combo_means.get(combo_key)
    if mean_abundance is None:
        # fallback: empty DataFrame
        return {}, pd.DataFrame(), pd.DataFrame(), {}
    top = mean_abundance.sort_values(ascending=False).head(top_n)
    all_top_microbes = top.index
    microbe_numbers = {microbe: f"M{idx+1}" for idx, microbe in enumerate(all_top_microbes)}
    comparison_data = {"Combo": mean_abundance.loc[all_top_microbes]}
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.index = [microbe_numbers[microbe] for microbe in comparison_df.index]
    id_mapping_df = pd.DataFrame({
        'Microbe ID': [microbe_numbers[microbe] for microbe in all_top_microbes],
        'Sequence': [microbe for microbe in all_top_microbes]
    })
    top_microbes = {"Combo": top}
    return top_microbes, comparison_df, id_mapping_df, microbe_numbers

top_microbes, comparison_df, id_mapping_df, microbe_numbers = get_top_microbes_from_combo_cache(
    cached_group_combo_means[group_col], selected_groups, top_n
)



# --- Enhanced Comparison Table ---
def create_comparison_table(cached_combo_means, selected_groups, top_n):
    comparison_data = {}
    for group in selected_groups:
        group_key = frozenset([group])
        mean_abundance = cached_combo_means.get(group_key, pd.Series(dtype='float64'))
        comparison_data[group] = mean_abundance

    comparison_df = pd.DataFrame(comparison_data)
    top_microbes = comparison_df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    comparison_df = comparison_df.loc[top_microbes]
    comparison_df.index.name = 'Microbe'
    return comparison_df

comparison_df = create_comparison_table(cached_group_combo_means[group_col], selected_groups, top_n)

# --- Update Comparison Table with Microbe Codes ---
def update_comparison_table_with_codes(comparison_df, microbe_numbers):
    updated_comparison_df = comparison_df.copy()
    updated_comparison_df.index = [microbe_numbers.get(microbe, microbe) for microbe in updated_comparison_df.index]
    return updated_comparison_df

comparison_df = update_comparison_table_with_codes(comparison_df, microbe_numbers)

# --- Visualize grouped bar chart using cached minimal data ---
st.header(f"Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s")
if not comparison_df.empty:
    fig, ax = plt.subplots(figsize=(max(8, len(comparison_df.index)*0.5), 6))
    comparison_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Mean Abundance')
    ax.set_xlabel('Microbe')
    ax.set_title(f"Comparison Across {group_label}s")
    ax.legend(title=group_label, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("No data available for the selected groups.")

# Per-group bar charts
st.header(f"Top {top_n} Microbes per {group_label}")
for group, microbes in top_microbes.items():
    st.subheader(f"{group_label if group_col == group else group}")
    # Only show top N for this group
    top_ids = [microbe_numbers[microbe] for microbe in microbes.index]
    fig, ax = plt.subplots()
    microbes.index = top_ids
    microbes.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel('Mean Abundance')
    ax.set_xlabel('Microbe (ID)')
    ax.set_title(f"{group}")
    st.pyplot(fig)
    st.write(microbes)

# Microbe ID mapping table
st.header("Microbe ID Mapping Table")
st.dataframe(id_mapping_df, use_container_width=True, hide_index=True)

# Comparison table
st.header(f"Comparison Table Across {group_label}s")
st.dataframe(comparison_df)

st.info("Upload your own files or change grouping column and top N for different comparisons.")
