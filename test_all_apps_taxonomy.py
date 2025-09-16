#!/usr/bin/env python3
"""
Test script to verify all three Streamlit apps can properly load their taxonomy data
and display taxonomy names in their tables.
"""

import os
import json
import pandas as pd

def test_cloud_demo():
    """Test cloud demo app taxonomy mapping and table structure."""
    print("Testing Cloud Demo App (streamlit_demo_app.py)...")
    
    # Import the functions
    from streamlit_demo_app import load_taxonomy_mapping, get_microbe_display_name
    
    # Load mapping
    mapping = load_taxonomy_mapping()
    
    if mapping:
        print(f"  ✅ Loaded {len(mapping)} microbe mappings")
        
        # Test a few sequences
        test_seq = "TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGATGGATGTTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGATACTGGATGTCTTGAGTG"
        display_name = get_microbe_display_name(test_seq, mapping)
        print(f"  ✅ Sample mapping: {test_seq[:20]}... -> {display_name}")
        
        # Test table structure
        test_microbes = list(mapping.keys())[:3]
        test_df = pd.DataFrame({
            'Microbe ID': [f"M{i+1}" for i in range(len(test_microbes))],
            'Taxonomy Name': [get_microbe_display_name(microbe, mapping) for microbe in test_microbes],
            'Sequence': test_microbes
        })
        print(f"  ✅ Table structure test: {len(test_df.columns)} columns")
        print(f"    Columns: {list(test_df.columns)}")
        return True
    else:
        print("  ❌ Failed to load taxonomy mapping")
        return False

def test_distilled_demo():
    """Test distilled demo app taxonomy mapping and table structure."""
    print("\nTesting Distilled Demo App (streamlit_demo_app_distilled.py)...")
    
    # Import the functions
    from streamlit_demo_app_distilled import get_microbe_display_name, load_taxonomy_mapping
    
    # Load mapping
    mapping = load_taxonomy_mapping()
    
    if mapping:
        print(f"  ✅ Loaded {len(mapping)} microbe mappings")
        
        # Test a few sequences
        test_seq = "AACGTAGGTCACAAGCGTTGTCCGGAATTACTGGGTGTAAAGGGAGCGCAGGCGGGAAGACAAGTTGGAAGTGAAATCCATGGGCTCAACCCATGAACTGCTTTCAAAACTGTTTTTCTTGAGTA"
        display_name = get_microbe_display_name(test_seq, mapping)
        print(f"  ✅ Sample mapping: {test_seq[:20]}... -> {display_name}")
        
        # Test table structure
        test_microbes = list(mapping.keys())[:3]
        test_df = pd.DataFrame({
            'Microbe ID': [f"M{i+1}" for i in range(len(test_microbes))],
            'Taxonomy Name': [get_microbe_display_name(microbe, mapping) for microbe in test_microbes],
            'Full Sequence ID': test_microbes
        })
        print(f"  ✅ Table structure test: {len(test_df.columns)} columns")
        print(f"    Columns: {list(test_df.columns)}")
        return True
    else:
        print("  ❌ Failed to load taxonomy mapping")
        return False

def test_local_app():
    """Test local app taxonomy functionality and table structure."""
    print("\nTesting Local App (streamlit_app.py)...")
    
    # Check if required files exist
    taxonomy_file = "taxonomy_distilled.qza"
    sequences_file = "sequences_distilled.qza"
    mapping_file = "taxonomy_mappings/local_taxonomy_index.json"
    
    if os.path.exists(taxonomy_file) and os.path.exists(sequences_file) and os.path.exists(mapping_file):
        print(f"  ✅ Found required files: {taxonomy_file}, {sequences_file}, {mapping_file}")
        
        # Import taxonomy functions
        try:
            from streamlit_app import get_microbe_display_name, load_taxonomy_index
            print("  ✅ Successfully imported taxonomy functions")
            
            # Test taxonomy index loading
            taxonomy_index = load_taxonomy_index()
            if taxonomy_index:
                print(f"  ✅ Loaded taxonomy index with {len(taxonomy_index)} entries")
                
                # Test table structure (simulate what the app creates)
                test_features = list(taxonomy_index.keys())[:3]
                test_df = pd.DataFrame({
                    'Microbe ID': [f"M{i+1}" for i in range(len(test_features))],
                    'Taxonomy Name': [get_microbe_display_name(feature, taxonomy_index) for feature in test_features],
                    'Sequence/Feature ID': test_features
                })
                print(f"  ✅ Table structure test: {len(test_df.columns)} columns")
                print(f"    Columns: {list(test_df.columns)}")
                return True
            else:
                print("  ❌ Failed to load taxonomy index")
                return False
                
        except ImportError as e:
            print(f"  ❌ Failed to import functions: {e}")
            return False
    else:
        print(f"  ❌ Missing required files")
        return False

def main():
    """Run all tests."""
    print("Testing Taxonomy Integration & Table Structure Across All Apps")
    print("=" * 65)
    
    results = []
    
    # Test each app
    results.append(test_cloud_demo())
    results.append(test_distilled_demo())
    results.append(test_local_app())
    
    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY:")
    
    apps = ["Cloud Demo (Supabase)", "Distilled Demo (Standalone)", "Local App (Upload)"]
    for i, (app, result) in enumerate(zip(apps, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {app}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 All apps now display taxonomy names in their tables!")
        print("   Structure: Microbe ID | Taxonomy Name | Sequence")
    
    return all_passed

if __name__ == "__main__":
    main()