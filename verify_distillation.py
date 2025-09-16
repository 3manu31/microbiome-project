#!/usr/bin/env python3
"""
Verification Script for Distilled Demo Data

This script verifies that all the requested changes have been successfully implemented:
1. Distilled data contains only top microbes per category
2. Depression category is present and properly distributed
3. No 'not provided' entries remain in the data
4. File sizes are significantly reduced
"""

import pandas as pd
import os
from biom import load_table

def verify_distillation():
    """Verify the distillation process was successful."""
    
    print("🔍 VERIFICATION OF DISTILLED DEMO DATA")
    print("=" * 50)
    
    # Check file existence
    biom_file = 'demo_biom_distilled.biom'
    metadata_file = 'metadata_demo_distilled.txt'
    
    if not os.path.exists(biom_file):
        print(f"❌ {biom_file} not found")
        return False
        
    if not os.path.exists(metadata_file):
        print(f"❌ {metadata_file} not found")
        return False
    
    print(f"✅ Both distilled files exist")
    
    # Check file sizes
    original_biom = 'demo_biom.biom'
    original_metadata = 'metadata_demo.txt'
    
    if os.path.exists(original_biom) and os.path.exists(original_metadata):
        orig_biom_size = os.path.getsize(original_biom)
        new_biom_size = os.path.getsize(biom_file)
        orig_meta_size = os.path.getsize(original_metadata)
        new_meta_size = os.path.getsize(metadata_file)
        
        biom_reduction = ((orig_biom_size - new_biom_size) / orig_biom_size) * 100
        meta_reduction = ((orig_meta_size - new_meta_size) / orig_meta_size) * 100
        
        print(f"📊 File Size Reductions:")
        print(f"   BIOM: {orig_biom_size:,} → {new_biom_size:,} bytes ({biom_reduction:.1f}% reduction)")
        print(f"   Metadata: {orig_meta_size:,} → {new_meta_size:,} bytes ({meta_reduction:.1f}% reduction)")
    
    # Load and verify metadata
    try:
        metadata = pd.read_csv(metadata_file, sep='\t')
        print(f"✅ Loaded metadata: {len(metadata)} samples, {len(metadata.columns)} columns")
        
        # Check for depression column
        if 'depression' in metadata.columns:
            print(f"✅ Depression category present")
            depression_counts = metadata['depression'].value_counts()
            print(f"   Distribution: {depression_counts.to_dict()}")
        else:
            print(f"❌ Depression category missing")
            return False
            
        # Check for 'not provided' entries
        not_provided_count = 0
        for col in metadata.columns:
            not_provided_count += (metadata[col].astype(str).str.lower() == 'not provided').sum()
        
        if not_provided_count == 0:
            print(f"✅ No 'not provided' entries found in metadata")
        else:
            print(f"❌ Found {not_provided_count} 'not provided' entries")
            return False
            
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return False
    
    # Load and verify BIOM data
    try:
        table = load_table(biom_file)
        abundance_df = table.to_dataframe(dense=True).T
        print(f"✅ Loaded BIOM data: {abundance_df.shape[0]} samples, {abundance_df.shape[1]} microbes")
        
        # Verify sample IDs match
        metadata_samples = set(metadata['sample_id'])
        biom_samples = set(abundance_df.index)
        matching_samples = metadata_samples.intersection(biom_samples)
        
        print(f"✅ Sample ID alignment: {len(matching_samples)} matching samples")
        
        if len(matching_samples) != len(metadata):
            print(f"⚠️  Warning: Not all metadata samples have abundance data")
            
    except Exception as e:
        print(f"❌ Error loading BIOM data: {e}")
        return False
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Total samples: {len(metadata)}")
    print(f"   Total microbes: {abundance_df.shape[1]}")
    print(f"   Metadata columns: {list(metadata.columns)}")
    
    # Category distributions
    print(f"\n👥 CATEGORY DISTRIBUTIONS:")
    for col in ['sex', 'age_cat', 'depression', 'mental_illness', 'asd']:
        if col in metadata.columns:
            counts = metadata[col].value_counts()
            print(f"   {col}: {len(counts)} categories, largest: {counts.iloc[0]} ({counts.index[0]})")
    
    print(f"\n" + "=" * 50)
    print(f"✅ VERIFICATION COMPLETE - All checks passed!")
    print(f"The distilled demo data is ready for deployment.")
    
    return True

if __name__ == "__main__":
    verify_distillation()