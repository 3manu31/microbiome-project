
"""
Microbiome Top Microbes Dashboard

This script loads sOTU abundance data from a .biom file and sample metadata, merges them, computes top microbes per group, and provides a clean starting point for a Streamlit dashboard. User can select grouping column for comparison.
"""
import pandas as pd
from biom import load_table

# --- Step 1: Load metadata ---
metadata_path = 'metadata.txt'  # Change to your actual metadata filename if needed
metadata = pd.read_csv(metadata_path, sep='\t')  # Use sep=',' if CSV

# --- Step 2: Load abundance data from .biom file ---
biom_path = 'deblur_125nt_no_blooms.biom'
table = load_table(biom_path)
abundance_df = table.to_dataframe(dense=True).T  # Samples as rows

# --- Step 3: Merge abundance and metadata ---
# Assume sample IDs are the index in abundance_df and a column in metadata
merged = abundance_df.merge(metadata, left_index=True, right_on='sample_id')

# --- Step 4: User selects grouping column ---
group_options = ['subset_healthy', 'mental_illness', 'sex', 'sample_type']
print("Available grouping columns:")
for i, col in enumerate(group_options):
    print(f"{i+1}. {col}")
choice = input("Select grouping column by number (1-4): ")
try:
    group_col = group_options[int(choice)-1]
except Exception:
    print("Invalid choice, defaulting to 'subset_healthy'.")
    group_col = 'subset_healthy'

# --- Step 5: Compute top microbes per group ---
def get_top_microbes(df, group_col, top_n=10):
    top_microbes = {}
    for group in df[group_col].dropna().unique():
        group_df = df[df[group_col] == group]
        mean_abundance = group_df.iloc[:, :-len(metadata.columns)].mean(axis=0)
        top = mean_abundance.sort_values(ascending=False).head(top_n)
        top_microbes[group] = top
    return top_microbes

top_microbes = get_top_microbes(merged, group_col)

# --- Step 6: Display results ---
for group, microbes in top_microbes.items():
    print(f"\nTop microbes for {group}:")
    print(microbes)

# --- Step 7: Ready for Streamlit integration ---
# You can now wrap this logic in a Streamlit app for interactive visualization.
