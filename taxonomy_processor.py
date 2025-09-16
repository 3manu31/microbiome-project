#!/usr/bin/env python3
"""
Taxonomy Processing Script for Microbiome Apps

This script handles taxonomy data processing for three different use cases:
1. Cloud demo app - precomputed mappings uploaded to Supabase
2. Distilled demo app - precomputed mappings stored locally
3. Local app - full taxonomy index for on-device processing
"""

import pandas as pd
import json
import os
from pathlib import Path
import biom
from typing import Dict, List, Optional, Tuple
import hashlib


class TaxonomyProcessor:
    def __init__(self):
        self.taxonomy_levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
    def parse_taxonomy_string(self, taxon_string: str) -> Dict[str, str]:
        """
        Parse QIIME2 taxonomy string into individual levels
        
        Args:
            taxon_string: String like "d__Bacteria;p__Bacteroidota;c__Bacteroidia;..."
            
        Returns:
            Dictionary with taxonomy levels
        """
        taxonomy = {}
        levels = taxon_string.split(';')
        
        level_mapping = {
            'd__': 'domain',
            'p__': 'phylum', 
            'c__': 'class',
            'o__': 'order',
            'f__': 'family',
            'g__': 'genus',
            's__': 'species'
        }
        
        for level in levels:
            level = level.strip()
            if len(level) < 3:
                continue
                
            prefix = level[:3]
            value = level[3:] if len(level) > 3 else ''
            
            if prefix in level_mapping:
                taxonomy[level_mapping[prefix]] = value if value else 'Unknown'
                
        # Fill in missing levels
        for level in self.taxonomy_levels:
            if level not in taxonomy:
                taxonomy[level] = 'Unknown'
                
        return taxonomy
    
    def load_taxonomy_data(self, taxonomy_file: str) -> pd.DataFrame:
        """Load and process taxonomy TSV file"""
        df = pd.read_csv(taxonomy_file, sep='\t')
        
        # Parse taxonomy strings
        taxonomy_parsed = []
        for _, row in df.iterrows():
            parsed = self.parse_taxonomy_string(row['Taxon'])
            parsed['feature_id'] = row['Feature ID']
            parsed['confidence'] = row['Confidence']
            parsed['full_taxonomy'] = row['Taxon']
            taxonomy_parsed.append(parsed)
            
        return pd.DataFrame(taxonomy_parsed)
    
    def load_biom_features(self, biom_file: str) -> List[str]:
        """Extract feature IDs from BIOM file"""
        table = biom.load_table(biom_file)
        return table.ids(axis='observation').tolist()
    
    def load_sequence_mapping(self, fasta_file: str) -> Dict[str, str]:
        """
        Load sequence to feature ID mapping from FASTA file
        Returns dict: {sequence: feature_id}
        """
        mapping = {}
        current_feature = None
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_feature = line[1:]  # Remove '>'
                elif current_feature and line:
                    mapping[line] = current_feature
                    
        return mapping
    
    def create_feature_taxonomy_mapping(self, taxonomy_df: pd.DataFrame, 
                                      biom_features: List[str],
                                      sequence_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Dict]:
        """
        Create mapping between BIOM feature IDs and taxonomy data
        
        For demo apps where we know the exact features in advance
        """
        mapping = {}
        
        # Create mapping for features that exist in both
        for biom_feature_id in biom_features:
            # If we have sequence mapping, convert sequence to feature_id
            if sequence_mapping and biom_feature_id in sequence_mapping:
                feature_id = sequence_mapping[biom_feature_id]
            else:
                feature_id = biom_feature_id
            
            # Try to find corresponding taxonomy
            taxonomy_match = taxonomy_df[taxonomy_df['feature_id'] == feature_id]
            
            if not taxonomy_match.empty:
                row = taxonomy_match.iloc[0]
                mapping[biom_feature_id] = {  # Use original BIOM feature ID as key
                    'feature_id': feature_id,
                    'domain': row['domain'],
                    'phylum': row['phylum'],
                    'class': row['class'],
                    'order': row['order'],
                    'family': row['family'],
                    'genus': row['genus'],
                    'species': row['species'],
                    'confidence': float(row['confidence']),
                    'full_taxonomy': row['full_taxonomy']
                }
            else:
                # If no direct match, create unknown entry
                mapping[biom_feature_id] = {
                    'feature_id': feature_id,
                    'domain': 'Unknown',
                    'phylum': 'Unknown',
                    'class': 'Unknown',
                    'order': 'Unknown',
                    'family': 'Unknown',
                    'genus': 'Unknown',
                    'species': 'Unknown',
                    'confidence': 0.0,
                    'full_taxonomy': 'Unknown'
                }
                
        return mapping
    
    def create_full_taxonomy_index(self, taxonomy_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create full taxonomy index for local app use
        
        This includes all taxonomy data for any potential feature
        """
        index = {}
        
        for _, row in taxonomy_df.iterrows():
            index[row['feature_id']] = {
                'domain': row['domain'],
                'phylum': row['phylum'],
                'class': row['class'],
                'order': row['order'],
                'family': row['family'],
                'genus': row['genus'],
                'species': row['species'],
                'confidence': float(row['confidence']),
                'full_taxonomy': row['full_taxonomy']
            }
            
        return index
    
    def generate_cloud_mapping(self, biom_file: str, taxonomy_file: str, 
                             fasta_file: str, output_file: str) -> str:
        """
        Generate precomputed taxonomy mapping for cloud demo app
        
        Returns cache key for Supabase storage
        """
        # Load data
        taxonomy_df = self.load_taxonomy_data(taxonomy_file)
        biom_features = self.load_biom_features(biom_file)
        sequence_mapping = self.load_sequence_mapping(fasta_file)
        
        # Create mapping
        mapping = self.create_feature_taxonomy_mapping(taxonomy_df, biom_features, sequence_mapping)
        
        # Generate cache key
        content_hash = hashlib.md5(
            f"{biom_file}_{taxonomy_file}".encode()
        ).hexdigest()
        cache_key = f"taxonomy_mapping_{content_hash}"
        
        # Save mapping
        output_data = {
            'cache_key': cache_key,
            'biom_file': os.path.basename(biom_file),
            'taxonomy_file': os.path.basename(taxonomy_file),
            'fasta_file': os.path.basename(fasta_file),
            'feature_count': len(mapping),
            'mapping': mapping
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Created cloud mapping: {output_file}")
        print(f"Features mapped: {len(mapping)}")
        print(f"Cache key: {cache_key}")
        
        return cache_key
    
    def generate_distilled_mapping(self, biom_file: str, taxonomy_file: str, 
                                 fasta_file: str, output_file: str) -> None:
        """
        Generate precomputed taxonomy mapping for distilled demo app
        """
        # Load data
        taxonomy_df = self.load_taxonomy_data(taxonomy_file)
        biom_features = self.load_biom_features(biom_file)
        sequence_mapping = self.load_sequence_mapping(fasta_file)
        
        # Create mapping
        mapping = self.create_feature_taxonomy_mapping(taxonomy_df, biom_features, sequence_mapping)
        
        # Save mapping
        output_data = {
            'biom_file': os.path.basename(biom_file),
            'taxonomy_file': os.path.basename(taxonomy_file),
            'fasta_file': os.path.basename(fasta_file),
            'feature_count': len(mapping),
            'mapping': mapping
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Created distilled mapping: {output_file}")
        print(f"Features mapped: {len(mapping)}")
    
    def generate_local_index(self, taxonomy_file: str, output_file: str) -> None:
        """
        Generate full taxonomy index for local app
        """
        # Load data
        taxonomy_df = self.load_taxonomy_data(taxonomy_file)
        
        # Create full index
        index = self.create_full_taxonomy_index(taxonomy_df)
        
        # Save index
        output_data = {
            'taxonomy_file': os.path.basename(taxonomy_file),
            'total_features': len(index),
            'index': index
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Created local taxonomy index: {output_file}")
        print(f"Total features indexed: {len(index)}")


def main():
    """Main execution function"""
    processor = TaxonomyProcessor()
    
    # File paths
    taxonomy_file = "taxonomy_distilled_exported/taxonomy.tsv"
    demo_biom = "demo_biom.biom" 
    distilled_biom = "demo_biom_distilled.biom"
    demo_fasta = "microbe_sequences_full_clean.fasta"  # For full demo
    distilled_fasta = "microbe_sequences_distilled.fasta"  # For distilled demo
    
    # Create output directory
    os.makedirs("taxonomy_mappings", exist_ok=True)
    
    print("=== Generating Taxonomy Mappings ===\n")
    
    # 1. Generate cloud demo mapping (for full demo data)
    if os.path.exists(demo_biom) and os.path.exists(demo_fasta):
        print("1. Generating cloud demo mapping...")
        cache_key = processor.generate_cloud_mapping(
            demo_biom, 
            taxonomy_file,
            demo_fasta,
            "taxonomy_mappings/cloud_demo_mapping.json"
        )
        print()
    
    # 2. Generate distilled demo mapping 
    if os.path.exists(distilled_biom) and os.path.exists(distilled_fasta):
        print("2. Generating distilled demo mapping...")
        processor.generate_distilled_mapping(
            distilled_biom,
            taxonomy_file,
            distilled_fasta, 
            "taxonomy_mappings/distilled_demo_mapping.json"
        )
        print()
    
    # 3. Generate local app index (full taxonomy)
    print("3. Generating local app taxonomy index...")
    processor.generate_local_index(
        taxonomy_file,
        "taxonomy_mappings/local_taxonomy_index.json"
    )
    print()
    
    print("=== Taxonomy Processing Complete ===")
    print("Generated files:")
    print("- taxonomy_mappings/cloud_demo_mapping.json (for cloud demo app)")
    print("- taxonomy_mappings/distilled_demo_mapping.json (for distilled demo app)")  
    print("- taxonomy_mappings/local_taxonomy_index.json (for local app)")


if __name__ == "__main__":
    main()