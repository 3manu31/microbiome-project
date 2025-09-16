#!/usr/bin/env python3
"""
Extract Microbe Sequences from BIOM Files to FASTA Format

This script extracts feature IDs (microbe sequences) from BIOM files and 
creates FASTA files for taxonomy assignment with QIIME2.
"""

import os
from biom import load_table

def extract_sequences_to_fasta(biom_file, output_fasta):
    """
    Extract feature IDs from BIOM file and write them as FASTA sequences.
    
    Args:
        biom_file (str): Path to the BIOM file
        output_fasta (str): Path to output FASTA file
    """
    print(f"Loading BIOM file: {biom_file}")
    
    try:
        # Load BIOM table
        table = load_table(biom_file)
        
        # Get feature IDs (these should be the actual DNA sequences)
        feature_ids = table.ids(axis='observation')
        
        print(f"Found {len(feature_ids)} features in BIOM file")
        
        # Write to FASTA file
        with open(output_fasta, 'w') as f:
            for i, feature_id in enumerate(feature_ids):
                # Use feature index as identifier and the feature_id as the sequence
                f.write(f">feature_{i+1}\n")
                f.write(f"{feature_id}\n")
        
        print(f"FASTA file created: {output_fasta}")
        print(f"Number of sequences: {len(feature_ids)}")
        
        # Show first few sequences for verification
        print("\nFirst 3 sequences (preview):")
        with open(output_fasta, 'r') as f:
            lines = f.readlines()
            for i in range(min(6, len(lines))):  # Show first 3 sequences (6 lines)
                print(lines[i].strip())
                
        # Show sequence length statistics
        sequence_lengths = []
        for feature_id in feature_ids:
            if isinstance(feature_id, str) and all(c in 'ATGCN' for c in feature_id.upper()):
                sequence_lengths.append(len(feature_id))
        
        if sequence_lengths:
            print(f"\nSequence length statistics:")
            print(f"  Min length: {min(sequence_lengths)} bp")
            print(f"  Max length: {max(sequence_lengths)} bp") 
            print(f"  Average length: {sum(sequence_lengths)/len(sequence_lengths):.1f} bp")
            print(f"  Valid DNA sequences: {len(sequence_lengths)} out of {len(feature_ids)}")
        else:
            print(f"\nWarning: No valid DNA sequences found. Feature IDs might be hash identifiers.")
            print("First 5 feature IDs:")
            for i, fid in enumerate(feature_ids[:5]):
                print(f"  {i+1}: {fid}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {biom_file}: {e}")
        return False

def main():
    """Main function to process both BIOM files."""
    
    # Define input BIOM files and output FASTA files
    files_to_process = [
        {
            'biom': 'demo_biom.biom',
            'fasta': 'microbe_sequences_full.fasta',
            'description': 'Full dataset'
        },
        {
            'biom': 'demo_biom_distilled.biom', 
            'fasta': 'microbe_sequences_distilled.fasta',
            'description': 'Distilled dataset'
        }
    ]
    
    print("🧬 Extracting Microbe Sequences from BIOM Files")
    print("=" * 60)
    
    for file_info in files_to_process:
        biom_file = file_info['biom']
        fasta_file = file_info['fasta']
        description = file_info['description']
        
        print(f"\n📁 Processing {description}")
        print(f"Input: {biom_file}")
        print(f"Output: {fasta_file}")
        
        if not os.path.exists(biom_file):
            print(f"❌ BIOM file not found: {biom_file}")
            continue
            
        success = extract_sequences_to_fasta(biom_file, fasta_file)
        
        if success:
            print(f"✅ Successfully created {fasta_file}")
        else:
            print(f"❌ Failed to create {fasta_file}")
        
        print("-" * 40)
    
    print("\n🎯 Next Steps:")
    print("1. Install QIIME2 if not already installed")
    print("2. Download appropriate classifier:")
    print("   - For 16S: wget https://data.qiime2.org/2024.2/common/silva-138-99-515-806-nb-classifier.qza")
    print("   - For ITS: wget https://data.qiime2.org/2024.2/common/unite-ver9-seqs-classifier.qza")
    print("3. Run taxonomy classification:")
    print("   qiime tools import --input-path microbe_sequences_distilled.fasta --output-path rep-seqs.qza --type 'FeatureData[Sequence]'")
    print("   qiime feature-classifier classify-sklearn --i-classifier silva-138-99-515-806-nb-classifier.qza --i-reads rep-seqs.qza --o-classification taxonomy.qza")

if __name__ == "__main__":
    main()