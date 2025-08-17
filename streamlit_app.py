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

st.title("Microbiome Top Microbes Dashboard")

# --- File uploaders ---
st.sidebar.header("Upload Your Files")
uploaded_metadata = st.sidebar.file_uploader("Upload metadata file (.txt or .csv)", type=["txt", "csv"])
uploaded_biom = st.sidebar.file_uploader("Upload biom file (.biom)", type=["biom"])

# --- Load metadata ---
try:
    if uploaded_metadata is not None:
        if hasattr(uploaded_metadata, 'size') and uploaded_metadata.size > 100 * 1024 * 1024:
            st.error("Uploaded metadata file is too large. Please upload a file smaller than 100MB.")
            st.stop()
        metadata = pd.read_csv(
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
        metadata = pd.read_csv('metadata_demo.txt', sep='\t', low_memory=False, encoding='utf-8')
except Exception as e:
    st.error(f"Error loading metadata: {e}")
    st.stop()

# --- Load abundance data from .biom file ---

def load_biom_file(uploaded_biom):
    try:
        content = uploaded_biom.read()
        # Try loading as JSON BIOM
        try:
            import json
            table = Table.from_json(json.loads(content.decode('utf-8')))
        except Exception:
            # If not JSON, save to temp file and load as HDF5
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp.flush()
                table = load_table(tmp.name)
        return table
    except Exception as e:
        st.error(f"Error loading biom file: {e}")
        st.stop()

try:
    if uploaded_biom is not None:
        if hasattr(uploaded_biom, 'size') and uploaded_biom.size > 100 * 1024 * 1024:
            st.error("Uploaded BIOM file is too large. Please upload a file smaller than 100MB.")
            st.stop()
        table = load_biom_file(uploaded_biom)
    else:
        # Load demo BIOM file
        if not os.path.exists('deblur_125nt_no_blooms.biom'):
            st.error("Demo BIOM file not found in repo. Please upload a BIOM file.")
            st.stop()
        with open('deblur_125nt_no_blooms.biom', 'rb') as demo_biom:
            class DummyUpload:
                def __init__(self, content):
                    self.content = content
                def read(self):
                    return self.content
            demo_biom_file = DummyUpload(demo_biom.read())
            table = load_biom_file(demo_biom_file)
except Exception as e:
    st.error(f"Error loading biom file: {e}")
    st.stop()

# --- Merge abundance and metadata ---


# Only limit samples/features for demo data
is_demo = (
    uploaded_metadata is None and uploaded_biom is None
)
MAX_FEATURES = 100 if is_demo else None

if metadata is None or table is None:
    st.warning("Please upload both a metadata file and a BIOM file to proceed.")
    st.stop()
try:
    abundance_df = table.to_dataframe(dense=True).T  # Samples as rows
    # Limit features for demo data only
    if is_demo:
        if MAX_FEATURES and abundance_df.shape[1] > MAX_FEATURES:
            abundance_df = abundance_df.iloc[:, :MAX_FEATURES]
    merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id')
except Exception as e:
    st.error(f"Error merging abundance and metadata: {e}. Please check that sample IDs match.")
    st.stop()

# --- Select grouping column and top N ---
group_options = [
    ('age_cat', 'Age Category'),
    ('mental_illness', 'Mental Illness'),
    ('sex', 'Sex'),
    ('sample_type', 'Sample Type'),
    ('asd', 'Autism Spectrum Disorder (ASD)')
]
group_col_label = st.selectbox("Select grouping column:", [label for _, label in group_options])
show_loading = False
if 'last_group_col_label' not in st.session_state:
    st.session_state['last_group_col_label'] = group_col_label
if group_col_label != st.session_state['last_group_col_label']:
    show_loading = True
    st.session_state['last_group_col_label'] = group_col_label
group_col = next(code for code, label in group_options if label == group_col_label)
group_label = group_col_label

# Get all unique group values for the selected column

import time

group_values = merged[group_col].dropna().unique().tolist()
default_selected = group_values.copy()

# Debounce logic: store last selection and time in session_state
if 'last_selected_groups' not in st.session_state:
    st.session_state['last_selected_groups'] = default_selected
if 'last_toggle_time' not in st.session_state:
    st.session_state['last_toggle_time'] = time.time()

selected_groups = st.multiselect(
    f"Show {group_label} options in grouped bar chart:",
    options=group_values,
    default=st.session_state['last_selected_groups'],
    help="Toggle which groups to display in the grouped bar chart."
)

now = time.time()
if selected_groups != st.session_state['last_selected_groups']:
    # Only update if 3 seconds have passed since last toggle
    if now - st.session_state['last_toggle_time'] < 3.0:
        st.warning("Please wait 3 seconds between toggles to avoid resource overload.")
    else:
        show_loading = True
        st.session_state['last_selected_groups'] = selected_groups
        st.session_state['last_toggle_time'] = now

if show_loading:
    st.info("Do not interact with the screen while content is loading...")
top_n = st.slider("Select number of top microbes:", min_value=5, max_value=15, value=10, step=1)

# --- Compute top microbes per group ---
def get_top_microbes(df, group_col, top_n=10):
    top_microbes = {}
    for group in df[group_col].dropna().unique():
        group_df = df[df[group_col] == group]
        mean_abundance = group_df.iloc[:, :-len(metadata.columns)].mean(axis=0)
        top = mean_abundance.sort_values(ascending=False).head(top_n)
        top_microbes[group] = top
    return top_microbes

top_microbes = get_top_microbes(merged, group_col, top_n)


# --- Find all unique top microbes across groups ---
all_top_microbes = pd.Index([])
for microbes in top_microbes.values():
    all_top_microbes = all_top_microbes.union(microbes.index)

# --- Assign a unique number to each microbe for tracking ---
microbe_numbers = {microbe: f"M{idx+1}" for idx, microbe in enumerate(all_top_microbes)}

# --- Prepare comparison table ---
comparison_data = {}
for group in top_microbes.keys():
    group_df = merged[merged[group_col] == group]
    mean_abundance = group_df.iloc[:, :-len(metadata.columns)].mean(axis=0)
    comparison_data[group] = mean_abundance.loc[all_top_microbes]
# Add overall mean abundance
overall_mean = merged.iloc[:, :-len(metadata.columns)].mean(axis=0).loc[all_top_microbes]
comparison_data['All'] = overall_mean
comparison_df = pd.DataFrame(comparison_data)

# --- Add microbe numbers to index for display (ID only for less crowding) ---

# Add microbe ID mapping table to the dashboard
comparison_df.index = [microbe_numbers[microbe] for microbe in comparison_df.index]
id_mapping_df = pd.DataFrame({
    'Microbe ID': [microbe_numbers[microbe] for microbe in all_top_microbes],
    'Sequence': [microbe for microbe in all_top_microbes]
})

# --- Visualize results ---


# Grouped Bar Chart: Microbe Abundance Across Groups (always at top)

# Filter comparison_df columns based on selected groups (keep 'All' column always)
filtered_columns = [col for col in comparison_df.columns if col in selected_groups or col == 'All']
filtered_comparison_df = comparison_df[filtered_columns]

st.header(f"Grouped Bar Chart: Microbe Abundance Across {group_label}s")
fig3, ax3 = plt.subplots(figsize=(max(8, len(filtered_comparison_df.index)*0.5), 6))
bar_width = 0.8 / len(filtered_comparison_df.columns)
indices = range(len(filtered_comparison_df.index))
for i, group in enumerate(filtered_comparison_df.columns):
    color = 'red' if group == 'All' else None
    ax3.bar([x + i*bar_width for x in indices], filtered_comparison_df[group], width=bar_width, label=group, color=color)
ax3.set_xticks([x + bar_width*(len(filtered_comparison_df.columns)/2-0.5) for x in indices])
ax3.set_xticklabels(filtered_comparison_df.index, rotation=90)
ax3.set_ylabel('Mean Abundance')
ax3.set_xlabel('Microbe (ID)')
ax3.legend()
st.pyplot(fig3)

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
