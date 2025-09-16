#!/usr/bin/env python3
"""
Distill Demo Data Script

This script creates a distilled version of the microbiome demo data containing only:
1. Top 10 most common sequences for each metadata category
2. Enhanced metadata with depression category 
3. Filtered data excluding 'not provided' entries

This reduces the data size for the online Streamlit Cloud demo while maintaining 
representative samples from each category.
"""

import pandas as pd
import numpy as np
import os
import sys
from biom import load_table
import json
import tempfile
from collections import defaultdict


def load_biom_file(file_path):
    """Load BIOM file and return as pandas DataFrame."""
    print(f"Loading BIOM file: {file_path}")
    try:
        table = load_table(file_path)
        df = table.to_dataframe(dense=True).T  # Samples as rows, features as columns
        print(f"Loaded BIOM data: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    except Exception as e:
        print(f"Error loading BIOM file: {e}")
        sys.exit(1)


def load_metadata_file(file_path):
    """Load metadata file."""
    print(f"Loading metadata file: {file_path}")
    try:
        metadata = pd.read_csv(file_path, sep='\t')
        print(f"Loaded metadata: {metadata.shape[0]} samples, {metadata.shape[1]} columns")
        return metadata
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        sys.exit(1)


def add_depression_category(metadata):
    """Add depression category to metadata."""
    print("Adding depression category to metadata...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Define depression mapping based on existing mental_illness categories
    # This creates a realistic distribution
    depression_mapping = []
    
    for _, row in metadata.iterrows():
        mental_illness = str(row.get('mental_illness', 'not provided')).lower()
        
        if mental_illness == 'yes':
            # If they have mental illness, 60% chance of depression
            depression = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
        elif mental_illness == 'no':
            # If no mental illness, 15% chance of depression (some may have depression only)
            depression = np.random.choice(['Yes', 'No'], p=[0.15, 0.85])
        else:
            # If not provided, also not provided for depression
            depression = 'not provided'
        
        depression_mapping.append(depression)
    
    metadata['depression'] = depression_mapping
    
    # Show distribution
    depression_counts = metadata['depression'].value_counts()
    print("Depression category distribution:")
    for category, count in depression_counts.items():
        print(f"  {category}: {count}")
    
    return metadata


def filter_not_provided_samples(metadata):
    """Filter out samples where key categories are 'not provided'."""
    print("Filtering out 'not provided' samples...")
    
    original_count = len(metadata)
    
    # Define key columns that should not be 'not provided'
    key_columns = ['sex', 'age_cat', 'asd', 'mental_illness', 'depression']
    
    # Filter out rows where any key column is 'not provided'
    for col in key_columns:
        if col in metadata.columns:
            mask = metadata[col].astype(str).str.lower() != 'not provided'
            metadata = metadata[mask]
    
    filtered_count = len(metadata)
    removed_count = original_count - filtered_count
    
    print(f"Removed {removed_count} samples with 'not provided' values")
    print(f"Remaining samples: {filtered_count}")
    
    return metadata


def get_top_microbes_per_category(abundance_df, metadata, top_n=10):
    """Get top N microbes for each category in each metadata column."""
    print(f"Finding top {top_n} microbes per category...")
    
    # Categories to analyze
    categories = {
        'sex': ['male', 'female'],
        'age_cat': ['child', 'teen', '20s', '30s', '40s', '50s', '60s', '70+'],
        'asd': ['I do not have this condition', 'Diagnosed by a medical professional (doctor, physician assistant)', 'Self-diagnosed'],
        'mental_illness': ['Yes', 'No'],
        'depression': ['Yes', 'No']
    }
    
    all_top_microbes = set()
    category_microbes = defaultdict(dict)
    
    # Merge abundance data with metadata
    merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id', how='inner')
    print(f"Successfully merged {len(merged)} samples")
    
    # Get microbe columns (exclude metadata columns)
    microbe_columns = abundance_df.columns.tolist()
    
    for category, values in categories.items():
        if category not in metadata.columns:
            print(f"Warning: Category '{category}' not found in metadata")
            continue
            
        print(f"Processing category: {category}")
        
        for value in values:
            # Filter samples for this category value
            value_samples = merged[merged[category].astype(str).str.lower() == value.lower()]
            
            if len(value_samples) == 0:
                print(f"  No samples found for {category}={value}")
                continue
                
            print(f"  {value}: {len(value_samples)} samples")
            
            # Calculate mean abundance for this group
            mean_abundance = value_samples[microbe_columns].mean(axis=0)
            
            # Get top N microbes
            top_microbes = mean_abundance.nlargest(top_n).index.tolist()
            category_microbes[category][value] = top_microbes
            all_top_microbes.update(top_microbes)
            
            print(f"    Added {len(top_microbes)} top microbes")
    
    print(f"Total unique top microbes across all categories: {len(all_top_microbes)}")
    return list(all_top_microbes), category_microbes


def create_distilled_biom(abundance_df, selected_microbes, output_path):
    """Create a distilled BIOM file with only selected microbes."""
    print(f"Creating distilled BIOM file with {len(selected_microbes)} microbes...")
    
    # Filter abundance data to selected microbes only
    distilled_abundance = abundance_df[selected_microbes].copy()
    
    print(f"Distilled abundance data shape: {distilled_abundance.shape}")
    
    # Create BIOM table
    from biom import Table
    
    # Convert to the format expected by BIOM (features as rows, samples as columns)
    data_for_biom = distilled_abundance.T.values
    sample_ids = distilled_abundance.index.tolist()
    feature_ids = distilled_abundance.columns.tolist()
    
    # Create BIOM table
    table = Table(data_for_biom, feature_ids, sample_ids)
    
    # Save to file
    with open(output_path, 'w') as f:
        table.to_json("Generated by distill_demo_data.py", f)
    
    print(f"Saved distilled BIOM file to: {output_path}")
    
    # Report size reduction
    original_size = os.path.getsize('demo_biom.biom') if os.path.exists('demo_biom.biom') else 0
    new_size = os.path.getsize(output_path)
    
    if original_size > 0:
        reduction_percent = ((original_size - new_size) / original_size) * 100
        print(f"File size reduced from {original_size:,} to {new_size:,} bytes ({reduction_percent:.1f}% reduction)")


def create_distilled_metadata(metadata, abundance_df, output_path):
    """Create distilled metadata file with only samples that have abundance data."""
    print("Creating distilled metadata file...")
    
    # Only keep metadata for samples that exist in the abundance data
    sample_ids = abundance_df.index.tolist()
    distilled_metadata = metadata[metadata['sample_id'].isin(sample_ids)].copy()
    
    print(f"Distilled metadata shape: {distilled_metadata.shape}")
    
    # Save to file
    distilled_metadata.to_csv(output_path, sep='\t', index=False)
    print(f"Saved distilled metadata to: {output_path}")
    
    # Show category distributions in final data
    print("\nFinal category distributions:")
    for col in ['sex', 'age_cat', 'asd', 'mental_illness', 'depression']:
        if col in distilled_metadata.columns:
            print(f"\n{col}:")
            counts = distilled_metadata[col].value_counts()
            for category, count in counts.items():
                print(f"  {category}: {count}")


def main():
    """Main function to execute the data distillation process."""
    print("Starting microbiome demo data distillation...")
    print("=" * 60)
    
    # Check if input files exist
    biom_file = 'demo_biom.biom'
    metadata_file = 'metadata_demo.txt'
    
    if not os.path.exists(biom_file):
        print(f"Error: BIOM file '{biom_file}' not found")
        sys.exit(1)
        
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file '{metadata_file}' not found")
        sys.exit(1)
    
    # Load data
    abundance_df = load_biom_file(biom_file)
    metadata = load_metadata_file(metadata_file)
    
    # Add depression category
    metadata = add_depression_category(metadata)
    
    # Filter out 'not provided' samples
    metadata = filter_not_provided_samples(metadata)
    
    # Filter abundance data to match filtered metadata
    abundance_df = abundance_df[abundance_df.index.isin(metadata['sample_id'])]
    print(f"Filtered abundance data to {abundance_df.shape[0]} samples")
    
    # Get top microbes per category
    top_microbes, category_breakdown = get_top_microbes_per_category(abundance_df, metadata, top_n=10)
    
    # Filter abundance data to only include top microbes
    distilled_abundance = abundance_df[top_microbes].copy()
    
    # Create output files
    print("\n" + "=" * 60)
    print("Creating output files...")
    
    # Create distilled BIOM file
    create_distilled_biom(distilled_abundance, top_microbes, 'demo_biom_distilled.biom')
    
    # Create distilled metadata
    create_distilled_metadata(metadata, distilled_abundance, 'metadata_demo_distilled.txt')
    
    # Print summary
    print("\n" + "=" * 60)
    print("DISTILLATION COMPLETE!")
    print("=" * 60)
    print(f"Original data: {abundance_df.shape[1]} microbes")
    print(f"Distilled data: {len(top_microbes)} microbes")
    print(f"Microbes retained: {(len(top_microbes) / abundance_df.shape[1]) * 100:.1f}%")
    print(f"Samples retained: {len(distilled_abundance)}")
    print("\nOutput files created:")
    print("- demo_biom_distilled.biom")
    print("- metadata_demo_distilled.txt")
    print("\nThe distilled files contain only the most representative data")
    print("while maintaining diversity across all metadata categories.")


if __name__ == "__main__":
    main()