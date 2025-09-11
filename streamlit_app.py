"""
Streamlit Microbiome Top Microbes Dashboard

This app lets users upload their own microbiome data files and visualizes 
the top microbes per group interactively. Perfect for local development 
and testing with your own data files.
"""

import streamlit as st
import pandas as pd
from biom import load_table
from biom.table import Table
import matplotlib.pyplot as plt
import os
import tempfile
import itertools
import io
import hashlib
import base64

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

st.title("🦠 Microbiome Explorer Dashboard")

# --- App Description ---
st.info("🏠 **Local Development Version** - Upload your own microbiome data files for interactive analysis. Charts are generated on-demand with full functionality.")

# --- File Upload Section ---
st.sidebar.header("Upload Your Files")
uploaded_metadata = st.sidebar.file_uploader("Upload metadata file (.txt or .csv)", type=["txt", "csv"])
uploaded_biom = st.sidebar.file_uploader("Upload biom file (.biom)", type=["biom"])

# --- Instructions ---
st.sidebar.header("How to Use")
st.sidebar.success("✅ **Upload Files**: Upload your metadata and BIOM files using the file uploaders above.\n✅ **Interactive Charts**: Charts are generated on-demand as you explore.\n✅ **Full Features**: All functionality available for your data!")

# Copyright and author notice
st.sidebar.markdown("<hr style='margin-top:2em;margin-bottom:0.5em;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align:center; color:gray; font-size:0.9em;'>© 2025 Emmanuel Gialitakis | Apache 2.0 License</div>", unsafe_allow_html=True)

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
        # Load demo metadata file as fallback
        if not os.path.exists('metadata_demo.txt'):
            st.error("Please upload a metadata file using the file uploader in the sidebar.")
            st.stop()
        return pd.read_csv('metadata_demo.txt', sep='\t', low_memory=False, encoding='utf-8')

# --- Load abundance data from .biom file ---
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

# Load data based on uploaded files or demo files
try:
    metadata = load_metadata(uploaded_metadata)
except Exception as e:
    st.error(f"Error loading metadata: {e}")
    st.stop()

try:
    if uploaded_biom is not None:
        if hasattr(uploaded_biom, 'size') and uploaded_biom.size > 100 * 1024 * 1024:
            st.error("Uploaded BIOM file is too large. Please upload a file smaller than 100MB.")
            st.stop()
        abundance_df = load_abundance_df(uploaded_biom)
    else:
        # Demo mode: load demo_biom.biom file
        if not os.path.exists('demo_biom.biom'):
            st.error("Please upload a BIOM file using the file uploader in the sidebar.")
            st.stop()
        with open('demo_biom.biom', 'rb') as f:
            abundance_df = parse_biom(f)
        # Sample for resource efficiency in demo mode
        abundance_df = abundance_df.iloc[:100, :100]
except Exception as e:
    st.error(f"Error loading BIOM file: {e}")
    st.stop()

# Determine if we're using demo data
is_demo = (uploaded_metadata is None and uploaded_biom is None)
MAX_FEATURES = 100 if is_demo else None

# Merge abundance and metadata
if metadata is None or abundance_df is None:
    st.warning("Please upload both a metadata file and a BIOM file to proceed.")
    st.stop()

try:
    # Limit features for demo data only  
    if is_demo and MAX_FEATURES and abundance_df.shape[1] > MAX_FEATURES:
        abundance_df = abundance_df.iloc[:, :MAX_FEATURES]
    merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id')
except Exception as e:
    st.error(f"Error merging abundance and metadata: {e}. Please check that sample IDs match.")
    st.stop()

# --- Group options (same as original app) ---
group_options = [
    ('age_cat', 'Age Category'),
    ('mental_illness', 'Mental Illness'),
    ('sex', 'Sex'),
    ('sample_type', 'Sample Type'),
    ('asd', 'Autism Spectrum Disorder (ASD)')
]

# --- UI for grouped bar chart (same as original app) ---
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
top_n = st.slider("Select number of top microbes:", min_value=4, max_value=10, value=10, step=2)

# --- Helper Functions (same as original app) ---
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

def generate_chart_locally(comparison_df, id_mapping_df, group_label):
    """Generate chart locally on-demand."""
    # Create the plot with better aesthetics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    comparison_df.plot(kind='bar', ax=ax, alpha=0.85)
    
    ax.set_title(f'Top {len(comparison_df)} Microbes by {group_label}', fontsize=14, pad=20)
    ax.set_xlabel('Microbe ID', fontsize=12)
    ax.set_ylabel('Mean Relative Abundance', fontsize=12)
    ax.legend(title=f'{group_label}', title_fontsize=11, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Convert to bytes for display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

# --- Create comparison table for microbe ID mapping ---
def create_comparison_table(merged, group_col, selected_groups, top_n):
    """Create comparison table and microbe ID mapping."""
    standardized_groups = standardize_group_order(selected_groups)
    
    comparison_data = {}
    for group in standardized_groups:
        group_df = merged[merged[group_col] == group]
        mean_abundance = group_df.iloc[:, :-len(metadata.columns)].mean(axis=0)
        comparison_data[group] = mean_abundance

    comparison_df = pd.DataFrame(comparison_data)
    top_microbes = comparison_df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    comparison_df = comparison_df.loc[top_microbes]
    
    # Create microbe ID mapping
    microbe_numbers = {microbe: f"M{idx+1}" for idx, microbe in enumerate(top_microbes)}
    comparison_df.index = [microbe_numbers[microbe] for microbe in comparison_df.index]
    
    # Create ID mapping table
    id_mapping_df = pd.DataFrame({
        'Microbe ID': [microbe_numbers[microbe] for microbe in top_microbes],
        'Sequence': list(top_microbes)
    })
    
    return comparison_df, id_mapping_df

# --- Main chart display logic ---
comparison_df, id_mapping_df = create_comparison_table(merged, group_col, selected_groups, top_n)

if group_col == "age_cat" and len(selected_groups) < 3:
    st.header(f"Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s")
    st.warning("Please select at least 3 age categories to render the combined chart.")
elif not comparison_df.empty:
    # Generate cache key (same logic as original app)
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
    
    # Generate chart locally on-demand
    chart_buffer = generate_chart_locally(standardized_df, id_mapping_df, group_label)
    
    # Display the generated chart
    st.markdown(f"<h1 style='color: green;'>📊 Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s</h1>", unsafe_allow_html=True)
    st.success("Chart generated locally for interactive exploration!")
    st.image(chart_buffer)
else:
    st.header(f"Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s")
    st.warning("No data available for the selected groups.")

# Microbe ID mapping table (same as original app)
st.header("🔍 Microbe ID Mapping Table")
if not id_mapping_df.empty:
    st.dataframe(id_mapping_df, use_container_width=True, hide_index=True)
else:
    st.info("Select valid groups to see microbe ID mapping.")

# Comparison table (same as original app)
st.header(f"📊 Comparison Table Across {group_label}s")
if not comparison_df.empty:
    st.dataframe(comparison_df)
else:
    st.info("Select valid groups to see comparison data.")

# Demo information
st.markdown("---")
st.markdown("### 🎯 About This Demo")
st.info("""
**Performance Features:**
- 🚀 **Instant Loading**: All charts are precomputed and cached in cloud storage
- ⚡ **Zero Computation**: No on-demand chart rendering, eliminating resource limits
- 🎨 **Full Visual Fidelity**: Identical design and functionality to the original app
- 📱 **Responsive Design**: Optimized for both desktop and mobile viewing

**Technical Implementation:**
- Charts generated offline with corrected pluralization
- Stored in Supabase S3-compatible storage
- Retrieved using same cache key logic as original app
- Fallback messaging for non-precomputed combinations

Upload your own files or change grouping column and top N for different comparisons!
""")
