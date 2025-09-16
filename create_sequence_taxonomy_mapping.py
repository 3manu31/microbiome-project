#!/usr/bin/env python3
"""
Create Proper Sequence to Taxonomy Mapping

This script creates the correct mapping between DNA sequences (from BIOM files)
and their taxonomy information by using the FASTA file as the bridge.
"""

import pandas as pd
import json
import biom
from pathlib import Path


def load_fasta_mapping(fasta_file):
    """Load sequence to feature_id mapping from FASTA file."""
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


def load_taxonomy_data(taxonomy_file):
    """Load taxonomy data."""
    df = pd.read_csv(taxonomy_file, sep='\t')
    
    taxonomy_dict = {}
    for _, row in df.iterrows():
        feature_id = row['Feature ID']
        taxon = row['Taxon']
        confidence = row['Confidence']
        
        # Parse taxonomy string
        taxonomy = parse_taxonomy_string(taxon)
        taxonomy['confidence'] = confidence
        taxonomy['full_taxonomy'] = taxon
        
        taxonomy_dict[feature_id] = taxonomy
        
    return taxonomy_dict


def parse_taxonomy_string(taxon_string):
    """Parse QIIME2 taxonomy string into individual levels."""
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
    taxonomy_levels = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    for level in taxonomy_levels:
        if level not in taxonomy:
            taxonomy[level] = 'Unknown'
            
    return taxonomy


def get_display_name(taxonomy_data):
    """Create a readable display name from taxonomy data."""
    # Try to build name from genus and species first
    genus = taxonomy_data.get('genus', '').strip()
    species = taxonomy_data.get('species', '').strip()
    
    if genus and genus != 'Unknown' and genus != '':
        if species and species != 'Unknown' and species != '':
            return f"{genus} {species}"
        else:
            return genus
    
    # Fallback to family
    family = taxonomy_data.get('family', '').strip()
    if family and family != 'Unknown' and family != '':
        return family
    
    # Fallback to higher levels
    for level in ['order', 'class', 'phylum', 'domain']:
        value = taxonomy_data.get(level, '').strip()
        if value and value != 'Unknown' and value != '':
            return value
    
    return 'Unknown'


def create_sequence_taxonomy_mapping():
    """Create the complete sequence to taxonomy mapping."""
    print("=== Creating Sequence to Taxonomy Mapping ===\n")
    
    # Load FASTA mapping (sequence -> feature_id)
    print("1. Loading FASTA sequence mapping...")
    fasta_mapping = load_fasta_mapping('microbe_sequences_distilled.fasta')
    print(f"   Found {len(fasta_mapping)} sequences")
    
    # Load taxonomy data (feature_id -> taxonomy)
    print("2. Loading taxonomy data...")
    taxonomy_data = load_taxonomy_data('taxonomy_distilled_exported/taxonomy.tsv')
    print(f"   Found {len(taxonomy_data)} taxonomy entries")
    
    # Load BIOM sequences
    print("3. Loading BIOM sequences...")
    table = biom.load_table('demo_biom_distilled.biom')
    biom_sequences = table.ids(axis='observation')
    print(f"   Found {len(biom_sequences)} sequences in BIOM")
    
    # Create final mapping
    print("4. Creating sequence to taxonomy mapping...")
    sequence_taxonomy_mapping = {}
    
    for sequence in biom_sequences:
        if sequence in fasta_mapping:
            feature_id = fasta_mapping[sequence]
            if feature_id in taxonomy_data:
                tax_data = taxonomy_data[feature_id]
                display_name = get_display_name(tax_data)
                
                sequence_taxonomy_mapping[sequence] = {
                    'feature_id': feature_id,
                    'display_name': display_name,
                    'taxonomy': tax_data
                }
                print(f"   ✅ {feature_id} -> {display_name}")
            else:
                print(f"   ❌ No taxonomy for {feature_id}")
        else:
            print(f"   ❌ Sequence not found in FASTA: {sequence[:50]}...")
    
    print(f"\n✅ Successfully mapped {len(sequence_taxonomy_mapping)}/{len(biom_sequences)} sequences")
    
    return sequence_taxonomy_mapping


def save_mappings(sequence_taxonomy_mapping):
    """Save the mappings in different formats for the apps."""
    
    # 1. Distilled demo mapping (simplified)
    print("\n5. Creating distilled demo mapping...")
    distilled_mapping = {}
    for sequence, data in sequence_taxonomy_mapping.items():
        distilled_mapping[sequence] = {
            'display_name': data['display_name'],
            'genus': data['taxonomy']['genus'],
            'species': data['taxonomy']['species'],
            'family': data['taxonomy']['family'],
            'confidence': data['taxonomy']['confidence']
        }
    
    distilled_output = {
        'biom_file': 'demo_biom_distilled.biom',
        'mapping_type': 'distilled_demo',
        'feature_count': len(distilled_mapping),
        'mapping': distilled_mapping
    }
    
    with open('taxonomy_mappings/distilled_demo_mapping.json', 'w') as f:
        json.dump(distilled_output, f, indent=2)
    print(f"   Saved distilled mapping: {len(distilled_mapping)} sequences")
    
    # 2. Cloud demo mapping (full taxonomy data)
    print("6. Creating cloud demo mapping...")
    cloud_mapping = {}
    for sequence, data in sequence_taxonomy_mapping.items():
        cloud_mapping[sequence] = {
            'display_name': data['display_name'],
            **data['taxonomy']  # Include all taxonomy levels
        }
    
    cloud_output = {
        'biom_file': 'demo_biom_distilled.biom',
        'mapping_type': 'cloud_demo',
        'feature_count': len(cloud_mapping),
        'cache_key': 'demo_taxonomy_mapping_v2',
        'mapping': cloud_mapping
    }
    
    with open('taxonomy_mappings/cloud_demo_mapping.json', 'w') as f:
        json.dump(cloud_output, f, indent=2)
    print(f"   Saved cloud mapping: {len(cloud_mapping)} sequences")
    
    # 3. Summary table for verification
    print("7. Creating summary table...")
    summary_data = []
    for sequence, data in sequence_taxonomy_mapping.items():
        summary_data.append({
            'Feature_ID': data['feature_id'],
            'Display_Name': data['display_name'],
            'Domain': data['taxonomy']['domain'],
            'Phylum': data['taxonomy']['phylum'],
            'Class': data['taxonomy']['class'],
            'Order': data['taxonomy']['order'],
            'Family': data['taxonomy']['family'],
            'Genus': data['taxonomy']['genus'],
            'Species': data['taxonomy']['species'],
            'Confidence': data['taxonomy']['confidence'],
            'Sequence': sequence[:50] + '...'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('taxonomy_mappings/sequence_taxonomy_summary.csv', index=False)
    print(f"   Saved summary table: {len(summary_data)} entries")
    
    return summary_df


def main():
    """Main execution."""
    # Create output directory
    Path('taxonomy_mappings').mkdir(exist_ok=True)
    
    # Create mapping
    sequence_taxonomy_mapping = create_sequence_taxonomy_mapping()
    
    if sequence_taxonomy_mapping:
        # Save mappings
        summary_df = save_mappings(sequence_taxonomy_mapping)
        
        print("\n" + "="*60)
        print("🧬 SEQUENCE TO TAXONOMY MAPPING COMPLETE!")
        print("="*60)
        print("\nSUMMARY OF MAPPED MICROBES:")
        print("-" * 60)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Feature_ID']:10} | {row['Display_Name']:25} | Conf: {row['Confidence']:.3f}")
        
        print(f"\n📁 Files created:")
        print(f"   - taxonomy_mappings/distilled_demo_mapping.json")
        print(f"   - taxonomy_mappings/cloud_demo_mapping.json") 
        print(f"   - taxonomy_mappings/sequence_taxonomy_summary.csv")
        
        print(f"\n✅ Ready for use in Streamlit apps!")
        
    else:
        print("\n❌ No mappings created. Check your input files.")


if __name__ == "__main__":
    main()