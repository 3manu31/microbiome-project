"""
Streamlit Microbiome Dashboard - Enhanced Demo with Button Control (Online Version)

This app supports both uploaded files and distilled demo data with enhanced features:
- Depression metadata category
- Filtered 'not provided' data
- Optimized for Streamlit Cloud deployment
- Supabase chart caching for performance
- Button-controlled chart generation for resource optimization
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
from biom import load_table, Table

# Configure page
st.set_page_config(
    page_title="Microbiome Explorer - Enhanced Demo",
    page_icon="🦠",
    layout="wide"
)

# Supabase Storage integration for cloud caching using S3-compatible API
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

st.title("🦠 Microbiome Explorer - Enhanced Demo")
st.info("⚡ **Enhanced Demo** - Supports distilled data | Depression category | Button-controlled charts | Supabase caching")

# Detect Streamlit Cloud environment
is_cloud = (
    os.environ.get("STREAMLIT_SERVER_HEADLESS", "false").lower() == "true" or
    os.environ.get("STREAMLIT_SHARING", "false").lower() == "true" or 
    "streamlit.io" in os.environ.get("STREAMLIT_SERVER_ADDRESS", "") or
    os.environ.get("HOSTNAME", "").startswith("streamlit") or
    "STREAMLIT_CLOUD" in os.environ
)

# Sidebar setup
st.sidebar.header("About This Demo")
if is_cloud:
    st.sidebar.success("☁️ **Cloud Mode**: Using distilled demo data\n✅ **Enhanced**: Depression category\n✅ **Optimized**: Button-controlled rendering")
else:
    st.sidebar.info("💻 **Local Mode**: Full functionality\n✅ **Upload**: Custom files supported\n✅ **Enhanced**: All features available")

st.sidebar.markdown("---")
st.sidebar.header("Limitations & Warnings")
st.sidebar.warning("⚠️ Cloud demo: Click 'Generate Charts' button\n⚠️ Toggle options one at a time\n⚠️ File upload disabled on cloud")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Emmanuel Gialitakis | Apache 2.0 License")

# Initialize caching systems
if 'chart_cache' not in st.session_state:
    st.session_state['chart_cache'] = {}
if 'cache_stats' not in st.session_state:
    st.session_state['cache_stats'] = {'hits': 0, 'misses': 0, 'supabase_hits': 0, 'supabase_misses': 0}

# Supabase S3-compatible configuration
@st.cache_resource
def get_s3_client():
    """Initialize S3-compatible client for Supabase Storage using access keys."""
    if not S3_AVAILABLE:
        return None
    
    try:
        supabase_url = st.secrets.get("SUPABASE_URL")
        access_key_id = st.secrets.get("ACCESS_KEY_ID")
        secret_access_key = st.secrets.get("SECRET_ACCESS_KEY")
        
        if supabase_url and access_key_id and secret_access_key:
            project_ref = supabase_url.split("//")[1].split(".")[0]
            s3_endpoint = f"https://{project_ref}.supabase.co/storage/v1/s3"
            
            return boto3.client(
                's3',
                endpoint_url=s3_endpoint,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name='us-east-1'
            )
    except Exception as e:
        st.warning(f"S3 connection failed: {e}")
    
    return None

# Initialize S3 client
s3_client = get_s3_client()

# File Upload Section (only for local runs)
if not is_cloud:
    st.sidebar.header("Upload Your Files")
    uploaded_metadata = st.sidebar.file_uploader("Upload metadata file (.txt or .csv)", type=["txt", "csv"])
    uploaded_biom = st.sidebar.file_uploader("Upload biom file (.biom)", type=["biom"])
else:
    uploaded_metadata = None
    uploaded_biom = None

# Load BIOM file functions
def parse_biom(uploaded_biom):
    """Parse BIOM file from uploaded content or file path."""
    if isinstance(uploaded_biom, str):
        # File path
        with open(uploaded_biom, 'rb') as f:
            content = f.read()
    else:
        # Uploaded file
        content = uploaded_biom.read()
    
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

# Load metadata with caching
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
        if not os.path.exists('metadata_demo_distilled.txt'):
            st.error("Demo metadata_demo_distilled.txt file not found. Please upload a metadata file.")
            st.stop()
        return pd.read_csv('metadata_demo_distilled.txt', sep='\t', low_memory=False, encoding='utf-8')

# Load data
try:
    metadata = load_metadata(uploaded_metadata)
except Exception as e:
    st.error(f"Error loading metadata: {e}")
    st.stop()

# Load BIOM file
try:
    if uploaded_biom is not None:
        if hasattr(uploaded_biom, 'size') and uploaded_biom.size > 100 * 1024 * 1024:
            st.error("Uploaded BIOM file is too large. Please upload a file smaller than 100MB.")
            st.stop()
        abundance_df = load_abundance_df(uploaded_biom)
    else:
        # Demo mode: load distilled BIOM file
        if not os.path.exists('demo_biom_distilled.biom'):
            st.error("Demo BIOM file demo_biom_distilled.biom not found.")
            st.stop()
        abundance_df = parse_biom('demo_biom_distilled.biom')
except Exception as e:
    st.error(f"Error loading biom file: {e}")
    st.stop()

# Merge abundance and metadata
if metadata is None or abundance_df is None:
    st.warning("Please upload both a metadata file and a BIOM file to proceed.")
    st.stop()

try:
    merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id')
    st.success(f"✅ Loaded {len(merged)} samples with {abundance_df.shape[1]} microbes")
except Exception as e:
    st.error(f"Error merging abundance and metadata: {e}. Please check that sample IDs match.")
    st.stop()

# Group options including depression
group_options = [
    ('age_cat', 'Age Category'),
    ('mental_illness', 'Mental Illness'),
    ('depression', 'Depression'),
    ('sex', 'Sex'),
    ('sample_type', 'Sample Type'),
    ('asd', 'Autism Spectrum Disorder (ASD)')
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

# UI for grouped bar chart
st.header("📊 Chart Configuration")

col1, col2 = st.columns([3, 1])

with col1:
    group_col_label = st.selectbox("Select grouping column:", [label for _, label in group_options])
    group_col = next(code for code, label in group_options if label == group_col_label)

with col2:
    top_n = st.slider("Number of top microbes:", min_value=4, max_value=10, value=10, step=2)

# Get available groups
group_values = [g for g in merged[group_col].dropna().unique().tolist()]
selected_groups = st.multiselect(
    f"Show {group_col_label} options in grouped bar chart:",
    options=group_values,
    default=group_values,
    help="Toggle which groups to display in the grouped bar chart."
)

# Add Generate Charts button for cloud version only
if is_cloud:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_charts = st.button(
            "🚀 Generate Charts", 
            type="primary", 
            use_container_width=True,
            help="Click to render charts with current settings. This prevents automatic re-rendering and optimizes resource usage."
        )
else:
    generate_charts = True  # Always true for local version

# Helper Functions
def standardize_group_order(groups):
    """Standardize the order of groups to ensure consistent chart rendering and caching."""
    def sort_key(item):
        item_str = str(item).lower()
        if item_str == "not provided":
            return "zzz_not_provided"  # Ensures it comes last
        if item_str[0].isalpha():
            return f"a_{item_str}"  # Letters get 'a_' prefix
        else:
            return f"b_{item_str}"  # Numbers/other get 'b_' prefix
    
    return sorted(groups, key=sort_key)

def create_comparison_table(cached_combo_means, selected_groups, top_n):
    """Create comparison table for multiple groups."""
    standardized_groups = standardize_group_order(selected_groups)
    
    comparison_data = {}
    for group in standardized_groups:
        group_key = frozenset([group])
        mean_abundance = cached_combo_means.get(group_key, pd.Series(dtype='float64'))
        comparison_data[group] = mean_abundance

    comparison_df = pd.DataFrame(comparison_data)
    top_microbes = comparison_df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    comparison_df = comparison_df.loc[top_microbes]
    comparison_df.index.name = 'Microbe'
    return comparison_df, top_microbes

def create_unified_microbe_numbers(all_microbe_sets):
    """Create a unified microbe numbering system across all charts."""
    all_unique_microbes = set()
    for microbe_set in all_microbe_sets:
        all_unique_microbes.update(microbe_set)
    
    sorted_microbes = sorted(list(all_unique_microbes))
    return {microbe: f"M{idx+1}" for idx, microbe in enumerate(sorted_microbes)}

def update_comparison_table_with_codes(comparison_df, microbe_numbers):
    updated_comparison_df = comparison_df.copy()
    updated_comparison_df = updated_comparison_df.rename(index={microbe: microbe_numbers.get(microbe, f"M{i+1}") 
                                                                for i, microbe in enumerate(updated_comparison_df.index)})
    return updated_comparison_df

# Supabase caching functions
def generate_cache_key(*args):
    """Generate a consistent, hashable cache key from arguments."""
    processed_args = []
    for arg in args:
        if isinstance(arg, list):
            processed_args.append(str(sorted(arg) if all(isinstance(x, (str, int, float)) for x in arg) else arg))
        elif isinstance(arg, dict):
            processed_args.append(str(sorted(arg.items())))
        else:
            processed_args.append(str(arg))
    
    key_string = '|'.join(processed_args)
    cache_key = hashlib.md5(key_string.encode()).hexdigest()
    return cache_key

def get_chart_from_supabase(cache_key):
    """Retrieve chart from Supabase storage using S3-compatible API."""
    if not s3_client:
        return None
    
    try:
        response = s3_client.get_object(Bucket='chart_cache', Key=f"{cache_key}.png")
        if response:
            st.session_state['cache_stats']['supabase_hits'] += 1
            return io.BytesIO(response['Body'].read())
    except ClientError:
        st.session_state['cache_stats']['supabase_misses'] += 1
    except Exception:
        st.session_state['cache_stats']['supabase_misses'] += 1
    
    return None

def save_chart_to_supabase(cache_key, chart_buffer):
    """Save chart to Supabase storage using S3-compatible API."""
    if not s3_client:
        return False
    
    try:
        chart_buffer.seek(0)
        s3_client.put_object(
            Bucket='chart_cache',
            Key=f"{cache_key}.png",
            Body=chart_buffer.getvalue(),
            ContentType='image/png'
        )
        return True
    except Exception as e:
        st.warning(f"Failed to save chart to Supabase: {e}")
        return False

def render_grouped_bar_chart(comparison_df, group_label, selected_groups):
    """Render grouped bar chart with caching."""
    standardized_groups = standardize_group_order(selected_groups)
    
    standardized_df = comparison_df.copy()
    if len(standardized_df.columns) > 1:
        available_cols = [col for col in standardized_groups if col in standardized_df.columns]
        standardized_df = standardized_df[available_cols]
    
    cache_key = generate_cache_key(
        "grouped", 
        group_label, 
        standardized_groups, 
        standardized_df.index.tolist(), 
        standardized_df.columns.tolist(),
        standardized_df.shape,
        str(standardized_df.values.tobytes())[:50]
    )
    
    cache_status = "newly_rendered"
    
    supabase_chart = get_chart_from_supabase(cache_key)
    if supabase_chart:
        cache_status = "supabase_cache"
        st.image(supabase_chart)
        return cache_status
    
    # Chart not in cache - render new one
    cache_status = "newly_rendered"
    fig, ax = plt.subplots(figsize=(max(8, len(standardized_df.index)*0.5), 6))
    standardized_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Mean Abundance')
    ax.set_xlabel('Microbe')
    ax.set_title(f"Comparison Across {group_label}s")
    ax.legend(title=group_label, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    
    # Save to Supabase cache
    save_chart_to_supabase(cache_key, buf)
    
    st.image(buf)
    plt.close(fig)
    
    return cache_status

# Only process and display charts when button is clicked (or always for local version)
if generate_charts:
    # Clear cache to prevent resource overload
    if 'last_interaction' not in st.session_state:
        st.session_state['last_interaction'] = None

    current_interaction = (group_col, tuple(selected_groups), top_n)
    if st.session_state['last_interaction'] != current_interaction:
        st.session_state['chart_cache'] = {}
        st.cache_data.clear()
        st.session_state['last_interaction'] = current_interaction

    # Calculate comparison data
    comparison_df, comparison_top_microbes = create_comparison_table(cached_group_combo_means[group_col], selected_groups, top_n)
    
    # Create unified microbe numbering system
    all_microbe_sets = [comparison_top_microbes]
    microbe_numbers = create_unified_microbe_numbers(all_microbe_sets)
    
    # Update comparison table with microbe codes
    comparison_df = update_comparison_table_with_codes(comparison_df, microbe_numbers)

    # Display results
    if selected_groups:
        if group_col == "age_cat" and len(selected_groups) < 3:
            st.header(f"Enhanced Grouped Bar Chart: Microbe Abundance Across {group_col_label}s")
            st.warning("Please select at least 3 age categories to render the combined chart.")
        elif not comparison_df.empty:
            # Pre-check cache status for header color
            standardized_groups = standardize_group_order(selected_groups)
            standardized_df = comparison_df.copy()
            if len(standardized_df.columns) > 1:
                available_cols = [col for col in standardized_groups if col in standardized_df.columns]
                standardized_df = standardized_df[available_cols]
            
            cache_key = generate_cache_key(
                "grouped", 
                group_col_label, 
                standardized_groups, 
                standardized_df.index.tolist(), 
                standardized_df.columns.tolist(),
                standardized_df.shape,
                str(standardized_df.values.tobytes())[:50]
            )
            
            if get_chart_from_supabase(cache_key):
                st.markdown(f"<h1 style='color: black;'>Enhanced Grouped Bar Chart: Microbe Abundance Across {group_col_label}s</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='color: navy;'>Enhanced Grouped Bar Chart: Microbe Abundance Across {group_col_label}s</h1>", unsafe_allow_html=True)
            
            render_grouped_bar_chart(comparison_df, group_col_label, selected_groups)
        else:
            st.header(f"Enhanced Grouped Bar Chart: Microbe Abundance Across {group_col_label}s")
            st.warning("No data available for the selected groups.")

        # Microbe ID mapping table
        st.header("Microbe ID Mapping Table")
        id_mapping_df = pd.DataFrame({
            'Microbe ID': [microbe_numbers.get(microbe, f"M{i+1}") for i, microbe in enumerate(comparison_top_microbes)],
            'Sequence': list(comparison_top_microbes)
        })
        st.dataframe(id_mapping_df, use_container_width=True, hide_index=True)

        # Comparison table
        st.header(f"Comparison Table Across {group_col_label}s")
        st.dataframe(comparison_df)

    else:
        st.info("👆 Please select group values to display a chart.")

else:
    # Show configuration preview when button hasn't been clicked (cloud only)
    if is_cloud:
        st.info("👆 **Click 'Generate Charts' button above to render visualizations with your selected settings.**")
        
        # Show current configuration preview
        st.header("📋 Current Configuration Preview")
        config_data = {
            "Setting": ["Grouping Category", "Selected Groups", "Top Microbes", "Total Samples"],
            "Value": [
                group_col_label,
                f"{len(selected_groups)} groups: {', '.join(selected_groups[:3])}{'...' if len(selected_groups) > 3 else ''}",
                f"{top_n} microbes",
                f"{len(merged)} samples"
            ]
        }
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        st.success("✅ **Resource Optimization**: Charts only render when you click the button, saving computational resources.")

# Footer info
st.markdown("---")
st.info("Upload your own files or change grouping column and top N for different comparisons.")

if is_cloud:
    st.info("💡 **Tip**: This online version uses button-controlled rendering to optimize performance on Streamlit Cloud.")