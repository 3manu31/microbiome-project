import pandas as pd

# Load the full metadata file
metadata = pd.read_csv('metadata.txt', sep='\t', low_memory=False)

# Keep only the required columns
columns_to_keep = [
    'sample_id', 'sex', 'age_cat', 'asd', 'sample_type', 'subset_healthy', 'mental_illness'
]
metadata_demo = metadata[columns_to_keep]

# Save the distilled metadata file
metadata_demo.to_csv('metadata_demo.txt', sep='\t', index=False)
print('Distilled metadata_demo.txt created.')
