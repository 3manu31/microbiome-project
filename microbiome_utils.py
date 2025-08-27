"""
Microbiome Analysis Utilities

Shared functions for microbiome data analysis across multiple dashboard implementations.
"""
import pandas as pd
import os


def get_top_microbes(df, group_col, metadata_columns_count, top_n=10):
    """
    Compute top microbes per group based on mean abundance.
    
    Args:
        df (pd.DataFrame): Merged dataframe with abundance data and metadata
        group_col (str): Column name to group by
        metadata_columns_count (int): Number of metadata columns to exclude from abundance calculation
        top_n (int): Number of top microbes to return per group
        
    Returns:
        dict: Dictionary mapping group names to Series of top microbes with their abundances
    """
    top_microbes = {}
    for group in df[group_col].dropna().unique():
        group_df = df[df[group_col] == group]
        # Calculate mean abundance excluding metadata columns
        mean_abundance = group_df.iloc[:, :-metadata_columns_count].mean(axis=0)
        top = mean_abundance.sort_values(ascending=False).head(top_n)
        top_microbes[group] = top
    return top_microbes


def load_metadata_safely(metadata_path, sep='\t'):
    """
    Safely load metadata file with proper error handling.
    
    Args:
        metadata_path (str): Path to metadata file
        sep (str): Separator for the file (default: '\t')
        
    Returns:
        pd.DataFrame: Loaded metadata
        
    Raises:
        SystemExit: If file cannot be loaded
    """
    # Check if file exists
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file '{metadata_path}' not found.")
        print("Please ensure the file exists or provide the correct path.")
        raise SystemExit(1)
    
    # Try to read the file
    try:
        metadata = pd.read_csv(metadata_path, sep=sep)
        print(f"Successfully loaded metadata from '{metadata_path}' with {len(metadata)} rows.")
        return metadata
    except FileNotFoundError:
        print(f"Error: Metadata file '{metadata_path}' not found.")
        raise SystemExit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Metadata file '{metadata_path}' is empty or contains no data.")
        raise SystemExit(1)
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse metadata file '{metadata_path}': {e}")
        print("Please check the file format and separator.")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: Unexpected error loading metadata file '{metadata_path}': {e}")
        raise SystemExit(1)


def get_metadata_path_from_args_or_env(default_path='metadata.txt'):
    """
    Get metadata path from command line arguments or environment variable.
    
    Args:
        default_path (str): Default path if no argument or env var provided
        
    Returns:
        str: Path to metadata file
    """
    import argparse
    import os
    
    # Check environment variable first
    env_path = os.environ.get('MICROBIOME_METADATA_PATH')
    if env_path:
        return env_path
    
    # Check command line arguments
    parser = argparse.ArgumentParser(description='Microbiome Dashboard')
    parser.add_argument('--metadata', '-m', default=default_path,
                       help='Path to metadata file (default: %(default)s)')
    args, _ = parser.parse_known_args()
    
    return args.metadata
