"""
Streamlit Microbiome Top Microbes Dashboard

This app lets users select a grouping column (healthy, mental illness, sex, sample type) and visualizes the top microbes per group interactively.
"""


import streamlit as st
import pandas as pd
from biom import load_table
import matplotlib.pyplot as plt
import os

st.title("Microbiome Top Microbes Dashboard")

# --- File uploaders ---
st.sidebar.header("Upload Your Files")
uploaded_metadata = st.sidebar.file_uploader("Upload metadata file (.txt or .csv)", type=["txt", "csv"])
uploaded_biom = st.sidebar.file_uploader("Upload biom file (.biom)", type=["biom"])

# --- Load metadata ---
try:
    if uploaded_metadata:
        metadata = pd.read_csv(uploaded_metadata, sep='\t' if uploaded_metadata.name.endswith('.txt') else ',')
    else:
        metadata_path = 'metadata.txt'
        if not os.path.exists(metadata_path):
            st.error("Metadata file not found. Please upload a metadata file.")
            st.stop()
        metadata = pd.read_csv(metadata_path, sep='\t')
except Exception as e:
    st.error(f"Error loading metadata: {e}")
    st.stop()

# --- Load abundance data from .biom file ---
try:
    if uploaded_biom:
        table = load_table(uploaded_biom)
    else:
        biom_path = 'deblur_125nt_no_blooms.biom'
        if not os.path.exists(biom_path):
            st.error("BIOM file not found. Please upload a biom file.")
            st.stop()
        table = load_table(biom_path)
    abundance_df = table.to_dataframe(dense=True).T  # Samples as rows
except Exception as e:
    st.error(f"Error loading biom file: {e}")
    st.stop()

# --- Merge abundance and metadata ---
try:
    merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id')
except Exception as e:
    st.error(f"Error merging abundance and metadata: {e}. Please check that sample IDs match.")
    st.stop()

# --- Select grouping column and top N ---
group_options = ['subset_healthy', 'mental_illness', 'sex', 'sample_type', 'asd']
group_labels = {
    'subset_healthy': 'Subset Healthy',
    'mental_illness': 'Mental Illness',
    'sex': 'Sex',
    'sample_type': 'Sample Type',
    'asd': 'Autism Spectrum Disorder (ASD)'
}
group_col = st.selectbox("Select grouping column:", group_options)
group_label = group_labels.get(group_col, group_col)
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
comparison_df.index = [microbe_numbers[microbe] for microbe in comparison_df.index]

# --- Visualize results ---

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

st.header(f"Comparison Table Across {group_label}s")
st.dataframe(comparison_df)


st.header(f"Grouped Bar Chart: Microbe Abundance Across {group_label}s")
fig3, ax3 = plt.subplots(figsize=(max(8, len(comparison_df.index)*0.5), 6))
bar_width = 0.8 / len(comparison_df.columns)
indices = range(len(comparison_df.index))
for i, group in enumerate(comparison_df.columns):
    color = 'red' if group == 'All' else None
    ax3.bar([x + i*bar_width for x in indices], comparison_df[group], width=bar_width, label=group, color=color)
ax3.set_xticks([x + bar_width*(len(comparison_df.columns)/2-0.5) for x in indices])
ax3.set_xticklabels(comparison_df.index, rotation=90)
ax3.set_ylabel('Mean Abundance')
ax3.set_xlabel('Microbe (ID)')
ax3.legend()
st.pyplot(fig3)

st.info("Upload your own files or change grouping column and top N for different comparisons.")
