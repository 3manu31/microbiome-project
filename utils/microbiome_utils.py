"""
Utility functions for microbiome data analysis.

This module contains shared functions for data loading, preprocessing,
and visualization that are used across both basic and advanced analysis scripts.

Author: Microbiome Analysis Project
Date: August 2025
"""

import pandas as pd
import numpy as np
import requests
import os
from typing import Tuple, Optional, Dict, List

def download_hmp_data(output_dir: str = "../data/raw", force_download: bool = False) -> str:
    """
    Download Human Microbiome Project abundance data.
    
    Args:
        output_dir: Directory to save downloaded data
        force_download: Whether to redownload if file exists
        
    Returns:
        Path to downloaded abundance table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # HMP abundance table URL (example - replace with actual HMP data URL)
    hmp_url = "https://www.hmpdacc.org/HMASM/abundance_table.txt"
    output_file = os.path.join(output_dir, "hmp_abundance_table.txt")
    
    if not os.path.exists(output_file) or force_download:
        print(f"Downloading HMP data to {output_file}...")
        try:
            response = requests.get(hmp_url, timeout=30)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            print("Download completed successfully")
            
        except requests.RequestException as e:
            print(f"Error downloading data: {e}")
            print("Creating sample data instead...")
            create_sample_hmp_data(output_file)
    else:
        print(f"Data already exists: {output_file}")
    
    return output_file

def create_sample_hmp_data(output_file: str) -> None:
    """
    Create sample HMP-style abundance data for demonstration.
    
    Args:
        output_file: Path to save sample data
    """
    print("Creating sample HMP abundance data...")
    
    # Generate realistic microbiome data
    np.random.seed(42)
    n_samples = 100
    taxa_names = [
        "k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides",
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Ruminococcaceae;g__Faecalibacterium",
        "k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Prevotellaceae;g__Prevotella",
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Ruminococcaceae;g__Ruminococcus",
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Eubacteriaceae;g__Eubacterium",
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Clostridiaceae;g__Clostridium",
        "k__Bacteria;p__Actinobacteria;c__Actinobacteria;o__Bifidobacteriales;f__Bifidobacteriaceae;g__Bifidobacterium",
        "k__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus",
        "k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacteriales;f__Enterobacteriaceae;g__Escherichia",
        "k__Bacteria;p__Verrucomicrobia;c__Verrucomicrobiae;o__Verrucomicrobiales;f__Verrucomicrobiaceae;g__Akkermansia"
    ]
    
    # Generate abundance matrix
    abundance_data = []
    sample_ids = [f"SRS{i:06d}" for i in range(1, n_samples + 1)]
    
    for i in range(n_samples):
        # Generate Dirichlet-distributed abundances
        alpha = np.random.gamma(2, 1, len(taxa_names))
        abundances = np.random.dirichlet(alpha)
        abundance_data.append(abundances)
    
    # Create DataFrame
    df = pd.DataFrame(abundance_data, index=sample_ids, columns=taxa_names)
    
    # Add sample metadata as comment at top
    header_lines = [
        "# Human Microbiome Project - Gut Microbiota Abundance Table",
        "# Sample abundance data for demonstration purposes",
        "# Taxonomic classification follows Greengenes format",
        "# Values represent relative abundances (sum to 1 per sample)",
        ""
    ]
    
    # Save to file
    with open(output_file, 'w') as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep='\t')
    
    print(f"Sample data created: {output_file}")

def load_abundance_table(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load microbiome abundance table from file.
    
    Args:
        file_path: Path to abundance table file
        
    Returns:
        Tuple of (abundance_dataframe, metadata_dict)
    """
    print(f"Loading abundance table from {file_path}...")
    
    # Read metadata from comment lines
    metadata = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                metadata[len(metadata)] = line.strip('# \n')
            else:
                break
    
    # Read abundance data
    abundance_df = pd.read_csv(file_path, sep='\t', index_col=0, comment='#')
    
    print(f"Loaded: {abundance_df.shape[0]} samples, {abundance_df.shape[1]} taxa")
    return abundance_df, metadata

def filter_low_abundance_taxa(abundance_df: pd.DataFrame, 
                             min_abundance: float = 0.001,
                             min_prevalence: float = 0.1) -> pd.DataFrame:
    """
    Filter out low-abundance and low-prevalence taxa.
    
    Args:
        abundance_df: Abundance DataFrame (samples x taxa)
        min_abundance: Minimum mean relative abundance
        min_prevalence: Minimum prevalence (fraction of samples)
        
    Returns:
        Filtered abundance DataFrame
    """
    print(f"Filtering taxa (min_abundance={min_abundance}, min_prevalence={min_prevalence})...")
    
    # Calculate mean abundance and prevalence
    mean_abundance = abundance_df.mean()
    prevalence = (abundance_df > 0).mean()
    
    # Filter criteria
    abundance_filter = mean_abundance >= min_abundance
    prevalence_filter = prevalence >= min_prevalence
    
    # Apply filters
    keep_taxa = abundance_filter & prevalence_filter
    filtered_df = abundance_df.loc[:, keep_taxa]
    
    print(f"Retained {filtered_df.shape[1]} / {abundance_df.shape[1]} taxa")
    return filtered_df

def clr_transform(abundance_df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Apply Centered Log-Ratio (CLR) transformation to abundance data.
    
    Args:
        abundance_df: Abundance DataFrame (samples x taxa)
        pseudocount: Small value to add to avoid log(0)
        
    Returns:
        CLR-transformed DataFrame
    """
    print("Applying CLR transformation...")
    
    # Add pseudocount
    data_with_pseudo = abundance_df + pseudocount
    
    # Calculate geometric mean for each sample
    geometric_mean = np.exp(np.log(data_with_pseudo).mean(axis=1))
    
    # CLR transformation
    clr_data = np.log(data_with_pseudo.div(geometric_mean, axis=0))
    
    return clr_data

def calculate_alpha_diversity(abundance_df: pd.DataFrame, 
                            metrics: List[str] = ['shannon', 'simpson', 'richness']) -> pd.DataFrame:
    """
    Calculate alpha diversity metrics.
    
    Args:
        abundance_df: Abundance DataFrame (samples x taxa)
        metrics: List of metrics to calculate
        
    Returns:
        DataFrame with diversity metrics
    """
    print(f"Calculating alpha diversity metrics: {metrics}")
    
    from scipy.stats import entropy
    
    diversity_results = []
    
    for sample in abundance_df.index:
        abundances = abundance_df.loc[sample].values
        
        # Remove zero abundances for calculations
        non_zero = abundances[abundances > 0]
        
        sample_metrics = {'SampleID': sample}
        
        if 'shannon' in metrics:
            # Shannon diversity
            shannon = entropy(non_zero, base=np.e)
            sample_metrics['Shannon'] = shannon
        
        if 'simpson' in metrics:
            # Simpson diversity (1 - Simpson's dominance)
            simpson_dominance = np.sum(non_zero ** 2)
            simpson = 1 - simpson_dominance
            sample_metrics['Simpson'] = simpson
        
        if 'richness' in metrics:
            # Species richness (number of observed taxa)
            richness = len(non_zero)
            sample_metrics['Richness'] = richness
        
        if 'evenness' in metrics:
            # Pielou's evenness
            if len(non_zero) > 1:
                evenness = entropy(non_zero, base=np.e) / np.log(len(non_zero))
            else:
                evenness = 0
            sample_metrics['Evenness'] = evenness
        
        diversity_results.append(sample_metrics)
    
    diversity_df = pd.DataFrame(diversity_results)
    
    print("Alpha diversity calculation completed")
    return diversity_df

def validate_abundance_data(abundance_df: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    """
    Validate that abundance data is properly formatted.
    
    Args:
        abundance_df: Abundance DataFrame to validate
        tolerance: Tolerance for sum-to-1 check
        
    Returns:
        True if data is valid, raises ValueError otherwise
    """
    print("Validating abundance data...")
    
    # Check for negative values
    if (abundance_df < 0).any().any():
        raise ValueError("Abundance data contains negative values")
    
    # Check if samples sum to 1 (relative abundances)
    sample_sums = abundance_df.sum(axis=1)
    if not np.allclose(sample_sums, 1.0, atol=tolerance):
        print("Warning: Sample abundances do not sum to 1.0")
        print(f"Sample sums range: {sample_sums.min():.6f} - {sample_sums.max():.6f}")
    
    # Check for all-zero samples
    zero_samples = (abundance_df.sum(axis=1) == 0).sum()
    if zero_samples > 0:
        raise ValueError(f"Found {zero_samples} samples with zero total abundance")
    
    # Check for all-zero taxa
    zero_taxa = (abundance_df.sum(axis=0) == 0).sum()
    if zero_taxa > 0:
        print(f"Warning: Found {zero_taxa} taxa with zero total abundance")
    
    print("Data validation completed successfully")
    return True

def parse_taxonomy_string(taxonomy_string: str) -> Dict[str, str]:
    """
    Parse Greengenes-style taxonomy string into components.
    
    Args:
        taxonomy_string: Taxonomy string (e.g., "k__Bacteria;p__Firmicutes;...")
        
    Returns:
        Dictionary with taxonomy levels
    """
    levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    prefixes = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    
    taxonomy_dict = {}
    
    # Split by semicolon
    parts = taxonomy_string.split(';')
    
    for i, part in enumerate(parts):
        if i < len(levels):
            # Remove prefix if present
            for prefix in prefixes:
                if part.startswith(prefix):
                    part = part[3:]  # Remove prefix
                    break
            
            taxonomy_dict[levels[i]] = part if part else 'Unknown'
    
    return taxonomy_dict

def get_genus_level_abundances(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate abundances to genus level.
    
    Args:
        abundance_df: Abundance DataFrame with full taxonomy names
        
    Returns:
        Genus-level abundance DataFrame
    """
    print("Aggregating abundances to genus level...")
    
    genus_abundances = {}
    
    for taxa in abundance_df.columns:
        # Parse taxonomy
        taxonomy = parse_taxonomy_string(taxa)
        genus = taxonomy.get('Genus', 'Unknown')
        
        if genus in genus_abundances:
            # Sum abundances for same genus
            genus_abundances[genus] += abundance_df[taxa]
        else:
            genus_abundances[genus] = abundance_df[taxa].copy()
    
    genus_df = pd.DataFrame(genus_abundances)
    
    print(f"Aggregated to {genus_df.shape[1]} genera")
    return genus_df

def save_results(data: pd.DataFrame, filepath: str, description: str = "") -> None:
    """
    Save analysis results to file with metadata.
    
    Args:
        data: DataFrame to save
        filepath: Output file path
        description: Optional description to include as header
    """
    print(f"Saving results to {filepath}...")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        if description:
            f.write(f"# {description}\n")
            f.write(f"# Generated on: {pd.Timestamp.now()}\n")
            f.write(f"# Shape: {data.shape[0]} rows x {data.shape[1]} columns\n")
            f.write("#\n")
        
        data.to_csv(f, sep='\t')
    
    print("Results saved successfully")
