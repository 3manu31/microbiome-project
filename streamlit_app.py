"""
Streamlit Microbiome Explorer Dashboard - GMRepo Dataset

This app analyzes microbiome data from the GMRepo database (PRJEB11419 project)
and visualizes the top microbes across different health conditions and demographics.
Features proper taxonomic classification at genus and species levels.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import hashlib
import numpy as np

# Local-only deployment check
def is_running_locally():
    """Check if running locally (not on cloud)"""
    try:
        # Check for cloud environment indicators
        cloud_indicators = (
            "STREAMLIT_SERVER_PORT" in os.environ or
            "STREAMLIT_CLOUD" in os.environ or
            os.environ.get("STREAMLIT_SERVER_HEADLESS", "false").lower() == "true" or
            os.environ.get("STREAMLIT_SHARING", "false").lower() == "true"
        )
        
        # Check if secrets are configured (cloud deployment)
        try:
            if hasattr(st, "secrets") and len(st.secrets) > 0:
                cloud_indicators = True
        except:
            # No secrets configured, likely local
            pass
            
        return not cloud_indicators
    except Exception:
        # If any error occurs, assume local
        return True

# Early exit if running on cloud
if not is_running_locally():
    st.error("🚫 **This app is designed for local development only.**")
    st.info("💡 For cloud deployment, please use `streamlit_demo_app.py` instead.")
    st.stop()

st.title("🦠 Microbiome Explorer Dashboard - GMRepo Dataset")

# GMRepo Data Loading Functions
@st.cache_data(show_spinner=False)
def load_gmrepo_metadata():
    """Load metadata from GMRepo dataset."""
    metadata_file = "gmrepo_data_PRJEB11419/all_runs_metadata.tsv"
    if os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file, sep='\t', low_memory=False)
        # Create age categories
        df['age_cat'] = pd.cut(df['host_age'], 
                              bins=[0, 18, 35, 50, 65, 100], 
                              labels=['<18', '18-35', '35-50', '50-65', '65+'],
                              include_lowest=True)
        return df
    else:
        st.error(f"GMRepo metadata file not found: {metadata_file}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_phenotype_mapping():
    """Load phenotype mapping from GMRepo dataset."""
    phenotypes_file = "gmrepo_data_PRJEB11419/prjeb11419_phenotypes.csv"
    if os.path.exists(phenotypes_file):
        return pd.read_csv(phenotypes_file)
    else:
        st.error(f"Phenotypes file not found: {phenotypes_file}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_phenotype_abundance_data(phenotype_name, taxonomic_level="species"):
    """Load abundance data for a specific phenotype."""
    file_name = f"{phenotype_name.replace(' ', '_')}_{taxonomic_level}_abundance.csv"
    file_path = f"gmrepo_data_PRJEB11419/phenotype_organized_data/{file_name}"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Filter out unknown/unidentified microbes to prevent incorrect aggregation
        # These represent different microbes that shouldn't be grouped together
        df = df[df['ncbi_taxon_id'] != -1]  # Remove entries with unknown taxon ID
        df = df[df['scientific_name'] != 'Unknown']  # Remove entries with unknown names
        
        return df
    else:
        st.warning(f"File not found: {file_path}")
        return pd.DataFrame()

def get_available_phenotypes():
    """Get list of available phenotypes based on files in the directory."""
    data_dir = "gmrepo_data_PRJEB11419/phenotype_organized_data"
    if not os.path.exists(data_dir):
        return []
    
    files = os.listdir(data_dir)
    phenotypes = set()
    
    for file in files:
        if file.endswith('_species_abundance.csv'):
            phenotype = file.replace('_species_abundance.csv', '').replace('_', ' ')
            phenotypes.add(phenotype)
    
    return sorted(list(phenotypes))

# Load data
metadata = load_gmrepo_metadata()
phenotype_mapping = load_phenotype_mapping()
available_phenotypes = get_available_phenotypes()

# Sidebar configuration
st.sidebar.header("📊 Analysis Configuration")

# Taxonomic level selector
taxonomic_level = st.sidebar.radio(
    "🔬 Taxonomic Level",
    ["Species", "Genus"],
    help="Species provides more specific microbial classification, while Genus gives broader taxonomic groups."
)

# Phenotype categories for organization
phenotype_categories = {
    "Demographics": ["Health"],
    "Mental Health": ["Depression", "Schizophrenia", "Bipolar Disorder", "Autism Spectrum Disorder"],
    "Gastrointestinal": ["Celiac Disease", "Constipation", "Inflammatory Bowel Diseases", 
                        "Clostridium Infections", "Diarrhea", "Irritable Bowel Syndrome", "Intestinal Diseases"],
    "Metabolic & Other": ["Diabetes Mellitus", "Autoimmune Diseases", "Cardiovascular Diseases"]
}

# Category and phenotype selection
category = st.sidebar.selectbox("📂 Condition Category", list(phenotype_categories.keys()))
available_in_category = [p for p in phenotype_categories[category] if p in available_phenotypes]

if not available_in_category:
    st.sidebar.error(f"No data available for {category} conditions")
    st.stop()

# Add "Compare All" button for the category
compare_all_clicked = st.sidebar.button(f"🚀 Compare All {category} Conditions")

if compare_all_clicked:
    # Set primary to first available condition and compare_conditions to all others
    primary_phenotype = available_in_category[0]
    compare_conditions = available_in_category[1:]
    st.sidebar.success(f"✅ Comparing all {len(available_in_category)} {category.lower()} conditions!")
    
    # Show optional "Add Healthy Controls" button only when comparing all
    if "Health" in available_phenotypes and "Health" not in available_in_category:
        add_healthy = st.sidebar.button("➕ Add Healthy Controls", 
                                       help="Include healthy control samples for comparison with all disease conditions")
        if add_healthy:
            compare_conditions.append("Health")
            st.sidebar.success("✅ Healthy controls added to comparison!")
    
else:
    primary_phenotype = st.sidebar.selectbox("🎯 Primary Condition", available_in_category)
    
    # Option to compare with other conditions
    compare_conditions = st.sidebar.multiselect(
        "🔄 Compare with additional conditions",
        options=[p for p in available_phenotypes if p != primary_phenotype],
        default=["Health"] if primary_phenotype != "Health" else [],
        help="Select additional conditions to compare in the analysis"
    )

# Combine primary and comparison conditions
selected_conditions = [primary_phenotype] + compare_conditions

# Number of top microbes to show

# Top N and ranking mode selector
top_n = st.sidebar.slider("🔝 Number of top microbes", min_value=5, max_value=20, value=10, step=1)
ranking_mode = st.sidebar.radio(
    "Ranking Mode",
    ["Top by Abundance", "Top by Abundance Difference (Outliers)"],
    help="Choose to see the most abundant or the most differentially abundant microbes between groups."
)

# Analysis type
analysis_type = st.sidebar.radio(
    "📈 Analysis Type",
    ["Condition Comparison", "Demographic Analysis"],
    help="Choose between comparing conditions or analyzing demographics within a condition"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** GMRepo Database (PRJEB11419)")
st.sidebar.markdown("**Total Samples:** 78,436 runs")
st.sidebar.markdown("**Healthy Controls:** 16,282 samples")

# Data processing and visualization functions
def create_abundance_comparison(selected_conditions, taxonomic_level, top_n):
    """Create comparison of microbe abundances across selected conditions."""
    all_data = []
    for condition in selected_conditions:
        abundance_data = load_phenotype_abundance_data(condition, taxonomic_level.lower())
        if abundance_data.empty:
            continue
        mean_abundance = abundance_data.groupby('scientific_name')['relative_abundance'].mean().reset_index()
        mean_abundance['condition'] = condition
        all_data.append(mean_abundance)
    if not all_data:
        return pd.DataFrame(), pd.DataFrame()
    combined_data = pd.concat(all_data, ignore_index=True)
    comparison_df = combined_data.pivot(index='scientific_name', columns='condition', values='relative_abundance')
    comparison_df = comparison_df.fillna(0)

    # Ranking logic
    if ranking_mode == "Top by Abundance" or len(comparison_df.columns) < 2:
        top_microbes = comparison_df.mean(axis=1).sort_values(ascending=False).head(top_n)
    else:
        # Outlier mode: top by max absolute difference between any two groups
        top_microbes = comparison_df.apply(lambda row: np.max(np.abs(row.values - row.values[:,None])), axis=1)
        top_microbes = top_microbes.sort_values(ascending=False).head(top_n)
    comparison_df = comparison_df.loc[top_microbes.index]

    # Create microbe ID mapping table
    id_mapping_data = []
    for idx, microbe_name in enumerate(comparison_df.index):
        microbe_id = f"M{idx+1}"
        id_mapping_data.append({
            'Microbe ID': microbe_id,
            'Scientific Name': microbe_name,
            'Mean Abundance': comparison_df.loc[microbe_name].mean()
        })
    id_mapping_df = pd.DataFrame(id_mapping_data)
    return comparison_df, id_mapping_df

def create_demographic_analysis(condition, demographic_type, taxonomic_level, top_n):
    """Analyze microbe abundance by demographics within a specific condition."""
    # Load abundance data for the condition
    abundance_data = load_phenotype_abundance_data(condition, taxonomic_level.lower())
    
    if abundance_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Merge with metadata to get demographic information
    merged_data = abundance_data.merge(metadata, left_on='run_id', right_on='run_id', how='inner')
    
    if demographic_type == "Age Groups":
        demographic_col = 'age_cat'
    elif demographic_type == "Sex":
        demographic_col = 'sex'
    else:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter out missing demographic data
    merged_data = merged_data.dropna(subset=[demographic_col])
    
    # Calculate mean abundance per microbe per demographic group
    demo_abundance = merged_data.groupby(['scientific_name', demographic_col], observed=True)['relative_abundance'].mean().reset_index()
    
    # Pivot to get microbes as rows, demographic groups as columns
    comparison_df = demo_abundance.pivot(index='scientific_name', columns=demographic_col, values='relative_abundance')
    comparison_df = comparison_df.fillna(0)
    
    # Get top microbes
    top_microbes = comparison_df.mean(axis=1).sort_values(ascending=False).head(top_n)
    comparison_df = comparison_df.loc[top_microbes.index]
    
    # Create ID mapping
    id_mapping_data = []
    for idx, microbe_name in enumerate(comparison_df.index):
        microbe_id = f"M{idx+1}"
        id_mapping_data.append({
            'Microbe ID': microbe_id,
            'Scientific Name': microbe_name,
            'Mean Abundance': top_microbes[microbe_name]
        })
    
    id_mapping_df = pd.DataFrame(id_mapping_data)
    
    return comparison_df, id_mapping_df

def generate_chart(comparison_df, chart_title):
    """Generate chart from comparison data."""
    if comparison_df.empty:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a color palette
    colors = None  # Let matplotlib use default colors
    
    comparison_df.plot(kind='bar', ax=ax, alpha=0.85, color=colors)
    
    ax.set_title(f'{chart_title}', fontsize=16, pad=20)
    ax.set_xlabel('Microbe (Scientific Name)', fontsize=12)
    ax.set_ylabel('Mean Relative Abundance (%)', fontsize=12)
    ax.legend(title='Condition/Group', title_fontsize=11, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Convert to bytes for display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

# Main analysis execution
if analysis_type == "Condition Comparison":
    st.header(f"🔬 Microbe Abundance Comparison - {taxonomic_level} Level")
    
    if len(selected_conditions) < 2:
        st.warning("Please select at least 2 conditions to compare.")
    else:
        comparison_df, id_mapping_df = create_abundance_comparison(selected_conditions, taxonomic_level, top_n)
        
        if not comparison_df.empty:
            chart_title = f"Top {top_n} {taxonomic_level} across Selected Conditions"
            chart_buffer = generate_chart(comparison_df, chart_title)
            
            if chart_buffer:
                st.image(chart_buffer)
                
                # Show sample sizes
                st.subheader("📊 Sample Sizes")
                sample_info = []
                for condition in selected_conditions:
                    phenotype_info = phenotype_mapping[phenotype_mapping['name'] == condition]
                    if not phenotype_info.empty:
                        sample_count = phenotype_info.iloc[0]['valid_runs']
                        sample_info.append({'Condition': condition, 'Sample Count': sample_count})
                
                if sample_info:
                    sample_df = pd.DataFrame(sample_info)
                    st.dataframe(sample_df, hide_index=True)

                    # Dynamic sample size analysis for all selected conditions
                    if len(sample_info) >= 2:
                        st.markdown("**📊 Sample Size Analysis:**")
                        
                        # Create a more comprehensive analysis
                        sample_df_display = sample_df.copy()
                        sample_df_display = sample_df_display.sort_values('Sample Count', ascending=False)
                        
                        # Calculate ratios relative to largest group
                        max_samples = sample_df_display['Sample Count'].max()
                        sample_df_display['Ratio'] = sample_df_display['Sample Count'].apply(
                            lambda x: f"1:{int(max_samples/x)}" if x > 0 else "N/A"
                        )
                        
                        # Display enhanced table
                        st.dataframe(sample_df_display[['Condition', 'Sample Count', 'Ratio']], hide_index=True)
                        
                        # Contextual message based on the comparison
                        min_samples = sample_df_display['Sample Count'].min()
                        max_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
                        
                        if max_ratio > 50:
                            st.warning("⚠️ **Large sample size imbalance** (>50:1 ratio). The group with fewer samples may show less reliable patterns. Consider this when interpreting differences.")
                        elif max_ratio > 10:
                            st.info("ℹ️ **Moderate sample size difference** (>10:1 ratio). This is common in disease vs. control studies and typically acceptable for analysis.")
                        else:
                            st.success("✅ **Well-balanced sample sizes** across all groups. Results should be reliable for all conditions.")
                        
                        # Special note for Health comparisons
                        health_in_comparison = any('health' in condition.lower() for condition in selected_conditions)
                        if health_in_comparison and len(selected_conditions) > 2:
                            st.info("💡 **Multi-group comparison tip**: With multiple conditions included, focus on patterns that are consistent across disease groups vs. healthy controls.")
                        
                    else:
                        st.info("Select multiple conditions to see sample size comparison.")
            
        else:
            st.warning("No data available for the selected conditions.")

else:  # Demographic Analysis
    st.header(f"👥 Demographic Analysis - {primary_phenotype}")
    
    demographic_type = st.selectbox(
        "Select demographic analysis",
        ["Age Groups", "Sex"]
    )
    
    comparison_df, id_mapping_df = create_demographic_analysis(primary_phenotype, demographic_type, taxonomic_level, top_n)
    
    if not comparison_df.empty:
        chart_title = f"Top {top_n} {taxonomic_level} in {primary_phenotype} by {demographic_type}"
        chart_buffer = generate_chart(comparison_df, chart_title)
        
        if chart_buffer:
            st.image(chart_buffer)
    else:
        st.warning(f"No demographic data available for {primary_phenotype}.")

# Display data tables
if 'comparison_df' in locals() and not comparison_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Microbe ID Mapping")
        if 'id_mapping_df' in locals() and not id_mapping_df.empty:
            st.dataframe(id_mapping_df, hide_index=True)
    
    with col2:
        st.subheader("📊 Abundance Data")
        # Round values for better display
        display_df = comparison_df.round(4)
        st.dataframe(display_df)

# Information panel
st.markdown("---")
st.markdown("### 🎯 About This Analysis")

# Add explanation of mean abundance
with st.expander("📚 Understanding Mean Relative Abundance", expanded=False):
    st.markdown("""
    **What does "Mean Relative Abundance %" mean?**
    
    🧬 **Relative Abundance**: For each sample, we calculate what percentage of the total microbiome each microbe represents.
    - Example: If a sample has 1000 total microbes and 50 are *Bacteroides vulgatus*, then *Bacteroides vulgatus* = 5% relative abundance
    
    📊 **Mean Across Samples**: We then average this percentage across all samples in each condition.
    - Example: If *Bacteroides vulgatus* is 5% in sample 1, 3% in sample 2, and 7% in sample 3, the mean = 5%
    
    🔍 **Important Note About Unknown Microbes**: 
    - ⚠️ **Original data**: The relative abundance percentages were calculated including unknown/unidentified microbes
    - 🧹 **Our filtering**: We exclude unknown microbes from analysis, but keep the original abundance percentages
    - 📈 **Impact**: This means percentages may not add up to 100% in our charts (typically 70-90%)
    - ✅ **Why this is OK**: We're comparing relative patterns between conditions, not absolute totals
    
    ⚖️ **Interpretation**:
    - **High values (>10%)**: Major components of the identified microbiome
    - **Medium values (1-10%)**: Important identified microbes  
    - **Low values (<1%)**: Minor but potentially significant identified microbes
    
    💡 **The numbers represent the average percentage of the total microbial community (including unknowns) that each identified microbe occupies in people with each condition.**
    """)

info_text = f"""
**Dataset Information:**
- 📊 **Source**: GMRepo Database, Project PRJEB11419
- 🔬 **Taxonomic Level**: {taxonomic_level}
- 🎯 **Analysis Type**: {analysis_type}
- 📈 **Top Microbes Shown**: {top_n}
- 🧹 **Data Filtering**: Unknown/unidentified microbes excluded to prevent aggregation artifacts

**Selected Conditions:**
{chr(10).join([f"- {condition}" for condition in selected_conditions])}

**Key Features:**
- ✅ **Proper Taxonomy**: Scientific names with NCBI taxon IDs
- ✅ **Large Scale**: Thousands of samples per condition
- ✅ **Comprehensive**: Covers major health conditions and controls
- ✅ **Interactive**: Real-time analysis based on your selections
- ✅ **Clean Data**: Filtered out unknown microbes that would skew results
"""

st.info(info_text)

# Copyright
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.9em;'>"
    "© 2025 Emmanuel Gialitakis | Apache 2.0 License | "
    "Data: GMRepo Database"
    "</div>", 
    unsafe_allow_html=True
)