"""
Streamlit Microbiome Top Microbes Dashboard

This app lets users select a grouping # --- File Upload# --- File Upload Section ---File Upload Section ---ex, sample type) and visualizes the top microbes per group interactively.
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

# Supabase Storage integration for cloud caching using S3-compatible API
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Supabase S3-compatible configuration (for cloud deployment only)
def get_s3_client():
    """Initialize S3-compatible client for Supabase Storage using access keys."""
    if not S3_AVAILABLE:
        return None
    
    # Only try to connect to Supabase in cloud environment
    if not is_cloud:
        return None
    
    try:
        supabase_url = st.secrets.get("SUPABASE_URL")
        access_key_id = st.secrets.get("ACCESS_KEY_ID")
        secret_access_key = st.secrets.get("SECRET_ACCESS_KEY")
        
        if supabase_url and access_key_id and secret_access_key:
            # Extract project reference from URL for S3 endpoint
            project_ref = supabase_url.split("//")[1].split(".")[0]
            s3_endpoint = f"https://{project_ref}.supabase.co/storage/v1/s3"
            
            return boto3.client(
                's3',
                endpoint_url=s3_endpoint,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name='us-east-1'  # Required but ignored by Supabase
            )
    except Exception as e:
        st.warning(f"S3 connection failed: {e}")
    
    return None

# Detect Streamlit Cloud environment
# Multiple checks to ensure reliable cloud detection
is_cloud = (
    os.environ.get("STREAMLIT_SERVER_HEADLESS", "false").lower() == "true" or
    os.environ.get("STREAMLIT_SHARING", "false").lower() == "true" or 
    "streamlit.io" in os.environ.get("STREAMLIT_SERVER_ADDRESS", "") or
    os.environ.get("HOSTNAME", "").startswith("streamlit") or
    "STREAMLIT_CLOUD" in os.environ
)


st.title("Microbiome Top Microbes Dashboard")

# Initialize S3 client early
s3_client = get_s3_client()

# --- Limitations & Warnings ---
st.sidebar.header("Limitations & Warnings")
st.sidebar.warning("\n- The live demo may be slow or crash if toggling options too quickly due to Streamlit Cloud resource limits.\n- Please toggle one option at a time and wait for the page to load before toggling again.\n- File upload is disabled on the cloud demo; to use this feature, install and run the app locally.\n- If you see errors or the app crashes, reload the page and try again.\n")

# Copyright and author notice
st.sidebar.markdown("<hr style='margin-top:2em;margin-bottom:0.5em;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align:center; color:gray; font-size:0.9em;'>© 2025 Emmanuel Gialitakis | Apache 2.0 License</div>", unsafe_allow_html=True)

# Initialize cache system early
if 'chart_cache' not in st.session_state:
    st.session_state['chart_cache'] = {}
if 'cache_stats' not in st.session_state:
    st.session_state['cache_stats'] = {'hits': 0, 'misses': 0, 'supabase_hits': 0, 'supabase_misses': 0}

chart_cache = st.session_state['chart_cache']
cache_stats = st.session_state['cache_stats']

# --- File Upload Section ---
# --- File uploaders (only enabled for local runs) ---
if not is_cloud:
    st.sidebar.header("Upload Your Files")
    uploaded_metadata = st.sidebar.file_uploader("Upload metadata file (.txt or .csv)", type=["txt", "csv"])
    uploaded_biom = st.sidebar.file_uploader("Upload biom file (.biom)", type=["biom"])
else:
    st.sidebar.header("Upload Your Files")
    st.sidebar.info("🚫 File upload is disabled on Streamlit Cloud. Using demo files instead.")
    st.sidebar.write("To upload your own files, run the app locally:")
    st.sidebar.code("streamlit run streamlit_app.py")
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
# Create a slider that shows actual microbe counts as labels
top_n = st.slider("Select number of top microbes:", min_value=4, max_value=10, value=10, step=2)


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



# --- Helper Functions ---

def standardize_group_order(groups):
    """
    Standardize the order of groups to ensure consistent chart rendering and caching.
    Sort letters first (a-z), then numbers (0-9), with "not provided" always at the end.
    """
    def sort_key(item):
        item_str = str(item).lower()
        if item_str == "not provided":
            return "zzz_not_provided"  # Ensures it comes last
        
        # Check if starts with letter vs number to prioritize letters
        if item_str[0].isalpha():
            return f"a_{item_str}"  # Letters get 'a_' prefix
        else:
            return f"b_{item_str}"  # Numbers/other get 'b_' prefix
    
    return sorted(groups, key=sort_key)

# --- Enhanced Comparison Table ---
def create_comparison_table(cached_combo_means, selected_groups, top_n):
    # Standardize group order for consistent table creation
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

comparison_df, comparison_top_microbes = create_comparison_table(cached_group_combo_means[group_col], selected_groups, top_n)

# --- Create unified microbe numbering system ---
def create_unified_microbe_numbers(all_microbe_sets):
    """Create a unified microbe numbering system across all charts."""
    # Combine all unique microbes from different sources
    all_unique_microbes = set()
    for microbe_set in all_microbe_sets:
        all_unique_microbes.update(microbe_set)
    
    # Sort for consistent numbering
    sorted_microbes = sorted(list(all_unique_microbes))
    return {microbe: f"M{idx+1}" for idx, microbe in enumerate(sorted_microbes)}

# Get all microbe sets that will be used
top_microbes, combo_comparison_df, id_mapping_df, _ = get_top_microbes_from_combo_cache(
    cached_group_combo_means[group_col], selected_groups, top_n
)

# Create unified numbering system
all_microbe_sets = [
    comparison_top_microbes,  # from comparison table
    id_mapping_df['Sequence'] if not id_mapping_df.empty else []  # from combo cache
]
microbe_numbers = create_unified_microbe_numbers(all_microbe_sets)

# --- Update Comparison Table with Microbe Codes ---
def update_comparison_table_with_codes(comparison_df, microbe_numbers):
    updated_comparison_df = comparison_df.copy()
    updated_comparison_df.index = [microbe_numbers.get(microbe, f"M{i+1}") for i, microbe in enumerate(updated_comparison_df.index)]
    return updated_comparison_df

comparison_df = update_comparison_table_with_codes(comparison_df, microbe_numbers)

# Update the ID mapping table with unified numbering
if not id_mapping_df.empty:
    id_mapping_df['Microbe ID'] = [microbe_numbers.get(seq, f"M{i+1}") for i, seq in enumerate(id_mapping_df['Sequence'])]

# --- Supabase caching functions ---

def generate_cache_key(*args):
    """Generate a consistent, hashable cache key from arguments."""
    # Convert all arguments to strings, ensuring consistent ordering for lists/dicts
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
        # Try to download the chart from Supabase storage
        response = s3_client.get_object(Bucket='chart_cache', Key=f"{cache_key}.png")
        if response:
            cache_stats['supabase_hits'] += 1
            return io.BytesIO(response['Body'].read())
    except ClientError as e:
        cache_stats['supabase_misses'] += 1
        # Chart not found in Supabase or error occurred
        pass
    except Exception as e:
        cache_stats['supabase_misses'] += 1
        pass
    
    return None

def save_chart_to_supabase(cache_key, chart_buffer):
    """Save chart to Supabase storage using S3-compatible API."""
    if not s3_client:
        return False
    
    try:
        # Upload chart to Supabase storage
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

def get_cached_chart(cache_key):
    """
    Get chart from cache with the following priority:
    1. Streamlit session state cache (fastest)
    2. Supabase storage (persistent)
    3. Return None if not found anywhere
    """
    # Check local cache first
    if cache_key in chart_cache:
        cache_stats['hits'] += 1
        return chart_cache[cache_key]
    
    # Check Supabase cache
    supabase_chart = get_chart_from_supabase(cache_key)
    if supabase_chart:
        # Store in local cache for faster future access
        chart_cache[cache_key] = supabase_chart
        return supabase_chart
    
    # Not found anywhere
    cache_stats['misses'] += 1
    return None

def save_chart_to_cache(cache_key, chart_buffer):
    """Save chart to both local cache and Supabase."""
    # Save to local cache
    chart_cache[cache_key] = chart_buffer
    
    # Save to Supabase for persistence
    if s3_client:
        save_chart_to_supabase(cache_key, chart_buffer)

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

def render_grouped_bar_chart(comparison_df, group_label, selected_groups):
    # Standardize group order for consistent rendering and caching
    standardized_groups = standardize_group_order(selected_groups)
    
    # Reorder DataFrame columns to match standardized order
    standardized_df = comparison_df.copy()
    if len(standardized_df.columns) > 1:  # Only reorder if multiple columns
        available_cols = [col for col in standardized_groups if col in standardized_df.columns]
        standardized_df = standardized_df[available_cols]
    
    # Generate consistent cache key including data shape and content
    cache_key = generate_cache_key(
        "grouped", 
        group_label, 
        standardized_groups, 
        standardized_df.index.tolist(), 
        standardized_df.columns.tolist(),
        standardized_df.shape,
        str(standardized_df.values.tobytes())[:50]  # Sample of data for uniqueness
    )
    
    # Track cache status for debugging
    cache_status = "newly_rendered"  # Default
    
    # Check local cache first
    if cache_key in chart_cache:
        cache_status = "in_app_cache"
        st.image(chart_cache[cache_key])
        return cache_status
    
    # Check Supabase cache
    supabase_chart = get_chart_from_supabase(cache_key)
    if supabase_chart:
        # Store in local cache for faster future access
        chart_cache[cache_key] = supabase_chart
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
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend on the right
    
    # Save chart to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    
    # Save to both caches
    save_chart_to_cache(cache_key, buf)
    
    st.image(buf)
    plt.close(fig)
    
    return cache_status


# Determine cache status first, then display header and chart
cache_status = None
if group_col == "age_cat" and len(selected_groups) < 3:
    st.header(f"Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s")
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
        group_label, 
        standardized_groups, 
        standardized_df.index.tolist(), 
        standardized_df.columns.tolist(),
        standardized_df.shape,
        str(standardized_df.values.tobytes())[:50]
    )
    
    # Determine cache status for header color
    if cache_key in chart_cache:
        cache_status = "in_app_cache"
        st.markdown(f"<h1 style='color: red;'>Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s</h1>", unsafe_allow_html=True)
    elif get_chart_from_supabase(cache_key):
        cache_status = "supabase_cache"
        st.markdown(f"<h1 style='color: black;'>Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s</h1>", unsafe_allow_html=True)
    else:
        cache_status = "newly_rendered"
        st.markdown(f"<h1 style='color: navy;'>Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s</h1>", unsafe_allow_html=True)
    
    render_grouped_bar_chart(comparison_df, group_label, selected_groups)
else:
    st.header(f"Enhanced Grouped Bar Chart: Microbe Abundance Across {group_label}s")
    st.warning("No data available for the selected groups.")

# Microbe ID mapping table
st.header("Microbe ID Mapping Table")
st.dataframe(id_mapping_df, use_container_width=True, hide_index=True)

# Comparison table
st.header(f"Comparison Table Across {group_label}s")
st.dataframe(comparison_df)

st.info("Upload your own files or change grouping column and top N for different comparisons.")
