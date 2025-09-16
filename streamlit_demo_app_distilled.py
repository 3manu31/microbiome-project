"""
Streamlit Microbiome Dashboard - Distilled Demo Version (Offline)

This app uses distilled demo data with:
- Only top microbes per category (97% size reduction)
- Enhanced metadata with depression category
- All 'not provided' data filtered out
- Optimized for local runs without Supabase
- Taxonomy integration for microbe identification
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import itertools
import hashlib
import io
import json
from biom import load_table, Table

# Configure page
st.set_page_config(
    page_title="Microbiome Explorer - Distilled Demo",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 Microbiome Explorer - Distilled Demo (Offline)")
st.info("⚡ **Offline Version**: Distilled data | Depression category | No 'not provided' entries | Local processing")

# Sidebar info
st.sidebar.header("About This Demo")
st.sidebar.success("✅ **Distilled Data**: 97% size reduction\n✅ **Enhanced**: Depression category added\n✅ **Clean**: No 'not provided' entries\n✅ **Local**: No cloud dependencies")
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Details:**")
st.sidebar.markdown("- 3,139 samples (from 9,511)")
st.sidebar.markdown("- 17 microbes (from 32,954)")
st.sidebar.markdown("- 8 metadata categories")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Emmanuel Gialitakis")

# Load BIOM file
@st.cache_data(show_spinner=False)
def load_biom_file(file_path):
    """Load BIOM file and return as pandas DataFrame."""
    try:
        table = load_table(file_path)
        df = table.to_dataframe(dense=True).T  # Samples as rows, features as columns
        return df
    except Exception as e:
        st.error(f"Error loading BIOM file: {e}")
        return None

# Load metadata
@st.cache_data(show_spinner=False)
def load_metadata(file_path):
    """Load metadata file."""
    try:
        return pd.read_csv(file_path, sep='\t', low_memory=False, encoding='utf-8')
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

# Load distilled demo data
@st.cache_data(show_spinner=False)
def load_demo_data():
    """Load the distilled demo data files."""
    biom_path = 'demo_biom_distilled.biom'
    metadata_path = 'metadata_demo_distilled.txt'
    
    if not os.path.exists(biom_path):
        st.error(f"Distilled BIOM file '{biom_path}' not found!")
        return None, None
        
    if not os.path.exists(metadata_path):
        st.error(f"Distilled metadata file '{metadata_path}' not found!")
        return None, None
    
    abundance_df = load_biom_file(biom_path)
    metadata = load_metadata(metadata_path)
    
    if abundance_df is None or metadata is None:
        return None, None
        
    return abundance_df, metadata

# Taxonomy Integration Functions
@st.cache_data(show_spinner=False)
def load_taxonomy_mapping():
    """Load precomputed taxonomy mapping for distilled demo data."""
    mapping_file = "taxonomy_mappings/distilled_demo_mapping.json"
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                data = json.load(f)
            return data.get('mapping', {})
        except Exception as e:
            st.warning(f"Could not load taxonomy mapping: {e}")
            return {}
    else:
        st.warning("Taxonomy mapping file not found. Microbe names will not be displayed.")
        return {}

def get_microbe_display_name(microbe_id, taxonomy_mapping):
    """Get display name for microbe using taxonomy."""
    if microbe_id in taxonomy_mapping:
        return taxonomy_mapping[microbe_id].get('display_name', microbe_id)
    
    # Fallback to microbe ID (truncated if it's a sequence)
    if len(microbe_id) > 20:
        return f"Microbe_{microbe_id[:8]}..."
    return microbe_id

# Load taxonomy mapping
taxonomy_mapping = load_taxonomy_mapping()
if taxonomy_mapping:
    st.success(f"✅ Loaded taxonomy data for {len(taxonomy_mapping)} microbes")
else:
    st.info("ℹ️ No taxonomy data available - microbe IDs will be displayed as-is")

# Load data
with st.spinner("Loading distilled demo data..."):
    abundance_df, metadata = load_demo_data()

if abundance_df is None or metadata is None:
    st.stop()

# Merge data
try:
    merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id', how='inner')
    st.success(f"✅ Loaded {len(merged)} samples with {abundance_df.shape[1]} microbes")
except Exception as e:
    st.error(f"Error merging data: {e}")
    st.stop()

# Group options including depression
group_options = [
    ('age_cat', 'Age Category'),
    ('mental_illness', 'Mental Illness'),
    ('depression', 'Depression'),
    ('sex', 'Sex'),
    ('asd', 'Autism Spectrum Disorder (ASD)'),
    ('sample_type', 'Sample Type')
]

# Precompute group combination means for caching
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
            combo_df = merged[merged[group_col].isin(combo)]
            mean_abundance = combo_df.iloc[:, :-len(metadata.columns)].mean(axis=0)
            cache[group_col][frozenset(combo)] = mean_abundance
        # Overall mean (all groups)
        overall_mean = merged.iloc[:, :-len(metadata.columns)].mean(axis=0)
        cache[group_col]['All'] = overall_mean
    return cache

cached_group_combo_means = precompute_group_combo_means(merged, group_options, metadata)

# UI for chart configuration
st.header("📊 Microbiome Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    group_col_label = st.selectbox(
        "Select grouping category:",
        [label for _, label in group_options],
        help="Choose how to group the samples for comparison"
    )
    
    group_col = next(code for code, label in group_options if label == group_col_label)

with col2:
    top_n = st.slider(
        "Top microbes to show:",
        min_value=5,
        max_value=17,  # Maximum available in distilled data
        value=10,
        step=1,
        help="Number of most abundant microbes to display"
    )

# Get available groups (excluding any remaining 'not provided' entries)
available_groups = [g for g in merged[group_col].dropna().unique().tolist() 
                   if str(g).lower() != 'not provided']

if len(available_groups) == 0:
    st.warning(f"No valid groups found for {group_col_label}")
    st.stop()

# Group selection
selected_groups = st.multiselect(
    f"Select {group_col_label.lower()} groups to compare:",
    options=available_groups,
    default=available_groups,
    help="Choose which groups to include in the comparison"
)

if not selected_groups:
    st.warning("Please select at least one group to display.")
    st.stop()

# Helper Functions
def standardize_group_order(groups):
    """Standardize the order of groups to ensure consistent chart rendering."""
    def sort_key(item):
        item_str = str(item).lower()
        if item_str == "not provided":
            return "zzz_not_provided"
        if item_str[0].isalpha():
            return f"a_{item_str}"
        else:
            return f"b_{item_str}"
    
    return sorted(groups, key=sort_key)

def create_comparison_table(cached_combo_means, selected_groups, top_n):
    """Create comparison table for multiple groups with taxonomy names."""
    standardized_groups = standardize_group_order(selected_groups)
    
    comparison_data = {}
    for group in standardized_groups:
        group_key = frozenset([group])
        mean_abundance = cached_combo_means.get(group_key, pd.Series(dtype='float64'))
        comparison_data[group] = mean_abundance

    comparison_df = pd.DataFrame(comparison_data)
    top_microbes = comparison_df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    comparison_df = comparison_df.loc[top_microbes]
    
    # Add taxonomy names to index
    if taxonomy_mapping:
        display_names = {}
        for microbe_id in comparison_df.index:
            display_name = get_microbe_display_name(microbe_id, taxonomy_mapping)
            display_names[microbe_id] = display_name
        
        comparison_df = comparison_df.rename(index=display_names)
    
    comparison_df.index.name = 'Microbe'
    return comparison_df, top_microbes

comparison_df, comparison_top_microbes = create_comparison_table(cached_group_combo_means[group_col], selected_groups, top_n)

# Create unified microbe numbering system
def create_unified_microbe_numbers(all_microbe_sets):
    """Create a unified microbe numbering system across all charts."""
    all_unique_microbes = set()
    for microbe_set in all_microbe_sets:
        all_unique_microbes.update(microbe_set)
    
    sorted_microbes = sorted(list(all_unique_microbes))
    return {microbe: f"M{idx+1}" for idx, microbe in enumerate(sorted_microbes)}

all_microbe_sets = [comparison_top_microbes]
microbe_numbers = create_unified_microbe_numbers(all_microbe_sets)

# Update comparison table with microbe codes
def update_comparison_table_with_codes(comparison_df, microbe_numbers):
    updated_comparison_df = comparison_df.copy()
    updated_comparison_df = updated_comparison_df.rename(index={microbe: microbe_numbers.get(microbe, f"M{i+1}") 
                                                                for i, microbe in enumerate(updated_comparison_df.index)})
    return updated_comparison_df

comparison_df = update_comparison_table_with_codes(comparison_df, microbe_numbers)

# Display results
if selected_groups:
    st.header(f"🔬 Top {top_n} Microbes by {group_col_label}")

    # Show sample counts
    st.subheader("📈 Sample Counts by Group")
    sample_counts = []
    for group in selected_groups:
        count = len(merged[merged[group_col] == group])
        sample_counts.append({"Group": group, "Sample Count": count})

    sample_counts_df = pd.DataFrame(sample_counts)
    st.dataframe(sample_counts_df, use_container_width=True, hide_index=True)

    # Create and display chart
    st.subheader("📊 Microbe Abundance Comparison")

    if not comparison_df.empty:
        fig, ax = plt.subplots(figsize=(max(10, len(comparison_df.index) * 0.6), 8))

        # Create grouped bar chart
        comparison_df.plot(kind='bar', ax=ax, width=0.8, colormap='viridis')

        ax.set_ylabel('Mean Abundance', fontsize=12)
        ax.set_xlabel('Microbe ID', fontsize=12)
        ax.set_title(f'Microbe Abundance Comparison Across {group_col_label}', fontsize=14, fontweight='bold')
        ax.legend(title=group_col_label, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Improve layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig)

        # Show comparison table
        st.subheader("📋 Detailed Abundance Data")
        st.dataframe(comparison_df.style.format("{:.6f}"), use_container_width=True)

        # Show microbe ID mapping
        st.subheader("🔍 Microbe ID Reference")
        mapping_df = pd.DataFrame({
            'Microbe ID': [microbe_numbers.get(microbe, f"M{i+1}") for i, microbe in enumerate(comparison_df.index)],
            'Taxonomy Name': [get_microbe_display_name(microbe, taxonomy_mapping) for microbe in comparison_df.index],
            'Full Sequence ID': comparison_df.index.tolist()
        })
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available for the selected groups.")

else:
    st.warning("Please select at least one group to display.")

# Show category distribution
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Category Distributions")
    for col_code, col_label in group_options:
        if col_code in metadata.columns:
            st.write(f"**{col_label}:**")
            counts = metadata[col_code].value_counts()
            for category, count in counts.head(8).items():  # Show top 8 categories
                st.write(f"  • {category}: {count}")
            if len(counts) > 8:
                st.write(f"  • ... and {len(counts) - 8} more")
            st.write("")

with col2:
    st.subheader("Data Summary")
    st.metric("Total Samples", len(metadata))
    st.metric("Total Microbes", len(abundance_df.columns))
    st.metric("Selected Groups", len(selected_groups))
    st.metric("Top Microbes Shown", top_n)
    
    # Data quality info
    st.write("**Data Quality:**")
    st.write("✅ All 'not provided' entries removed")
    st.write("✅ Only top microbes per category included")
    st.write("✅ Depression category added")

# Footer
st.markdown("---")
st.markdown("**About the Data**: This demo uses a distilled subset of microbiome data optimized for fast loading and analysis. The original dataset was reduced by 97% while preserving the most important microbes from each category.")

st.markdown("**New Features**: Depression category has been added as a metadata field for enhanced psychological health analysis.")

st.info("This is the offline version with local processing only. No cloud dependencies required.")