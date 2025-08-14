"""
Data Loader for Microbiome Analysis

This script provides functions to load and prepare microbiome datasets
for both basic and advanced analysis workflows.

Author: Microbiome Analysis Project
Date: August 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import requests
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def download_sample_data(output_dir: str = "data/raw") -> None:
    """
    Download or create sample microbiome datasets.
    
    Args:
        output_dir: Directory to save data files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 Preparing sample microbiome datasets...")
    
    # Create basic analysis dataset (HMP-style)
    create_hmp_sample_data(os.path.join(output_dir, "hmp_gut_abundance.txt"))
    
    # Create advanced analysis dataset (with disease labels)
    create_disease_sample_data(os.path.join(output_dir, "disease_abundance.txt"),
                              os.path.join(output_dir, "disease_metadata.txt"))
    
    print("✅ Sample datasets created successfully!")

def create_hmp_sample_data(output_file: str) -> None:
    """Create HMP-style gut microbiome abundance data."""
    print(f"   Creating HMP sample data: {output_file}")
    
    np.random.seed(42)
    n_samples = 100
    
    # Realistic gut microbiome taxa
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
        "k__Bacteria;p__Verrucomicrobia;c__Verrucomicrobiae;o__Verrucomicrobiales;f__Verrucomicrobiaceae;g__Akkermansia",
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Roseburia",
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Coprococcus",
        "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Blautia",
        "k__Bacteria;p__Firmicutes;c__Negativicutes;o__Selenomonadales;f__Veillonellaceae;g__Dialister",
        "k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Porphyromonadaceae;g__Parabacteroides"
    ]
    
    sample_ids = [f"SRS{i:06d}" for i in range(1, n_samples + 1)]
    
    # Generate abundance matrix using Dirichlet distribution
    abundance_data = []
    for i in range(n_samples):
        alpha = np.random.gamma(2, 1, len(taxa_names))
        abundances = np.random.dirichlet(alpha)
        abundance_data.append(abundances)
    
    # Create DataFrame
    df = pd.DataFrame(abundance_data, index=sample_ids, columns=taxa_names)
    
    # Save with header
    with open(output_file, 'w') as f:
        f.write("# Human Microbiome Project - Gut Microbiota Abundance Table\n")
        f.write("# Sample abundance data for demonstration purposes\n")
        f.write("# Taxonomic classification follows Greengenes format\n")
        f.write("# Values represent relative abundances (sum to 1 per sample)\n")
        f.write("#\n")
        df.to_csv(f, sep='\t')

def create_disease_sample_data(abundance_file: str, metadata_file: str) -> None:
    """Create disease vs healthy microbiome dataset."""
    print(f"   Creating disease dataset: {abundance_file}, {metadata_file}")
    
    np.random.seed(42)
    n_samples = 200
    n_taxa = 30
    
    # Taxa names
    taxa_names = [
        "Bacteroides_vulgatus", "Faecalibacterium_prausnitzii", "Prevotella_copri",
        "Ruminococcus_bromii", "Eubacterium_rectale", "Clostridium_butyricum",
        "Bifidobacterium_longum", "Lactobacillus_rhamnosus", "Enterobacteriaceae_sp",
        "Akkermansia_muciniphila", "Roseburia_intestinalis", "Coprococcus_eutactus",
        "Blautia_obeum", "Dialister_invisus", "Parabacteroides_distasonis",
        "Alistipes_putredinis", "Oscillospira_sp", "Sutterella_wadsworthensis",
        "Collinsella_aerofaciens", "Dorea_longicatena"
    ] + [f"Taxa_{i:02d}" for i in range(21, n_taxa + 1)]
    
    # Create sample IDs and disease labels
    sample_ids = [f"Sample_{i:03d}" for i in range(1, n_samples + 1)]
    disease_labels = ["Healthy"] * int(n_samples * 0.6) + ["IBS"] * int(n_samples * 0.4)
    np.random.shuffle(disease_labels)
    
    # Generate abundance data with disease-specific patterns
    abundance_matrix = []
    for label in disease_labels:
        if label == "Healthy":
            # Healthy microbiome
            alpha = np.random.gamma(2, 1, n_taxa)
            alpha[1] *= 2    # More Faecalibacterium
            alpha[9] *= 1.5  # More Akkermansia
        else:
            # IBS microbiome (dysbiosis)
            alpha = np.random.gamma(1.5, 0.8, n_taxa)
            alpha[0] *= 1.5  # More Bacteroides
            alpha[8] *= 2    # More Enterobacteriaceae
            alpha[1] *= 0.5  # Less Faecalibacterium
            alpha[9] *= 0.7  # Less Akkermansia
        
        abundances = np.random.dirichlet(alpha)
        abundance_matrix.append(abundances)
    
    # Create abundance DataFrame
    abundance_df = pd.DataFrame(abundance_matrix, index=sample_ids, columns=taxa_names)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Disease_Status': disease_labels,
        'Age': np.random.normal(45, 15, n_samples).astype(int),
        'BMI': np.random.normal(25, 4, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Antibiotics_Recent': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    })
    
    # Save files
    abundance_df.to_csv(abundance_file, sep='\t')
    metadata_df.to_csv(metadata_file, sep='\t', index=False)

def load_basic_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for basic analysis.
    
    Returns:
        Tuple of (abundance_dataframe, metadata_dataframe)
    """
    data_dir = "data/raw"
    abundance_file = os.path.join(data_dir, "hmp_gut_abundance.txt")
    
    if not os.path.exists(abundance_file):
        print("📥 Data not found. Downloading sample data...")
        download_sample_data()
    
    print(f"📊 Loading basic analysis data from {abundance_file}")
    
    # Read abundance data
    abundance_df = pd.read_csv(abundance_file, sep='\t', index_col=0, comment='#')
    
    # Create simple metadata
    metadata_df = pd.DataFrame({
        'SampleID': abundance_df.index,
        'Subject': [f"Subject_{i//2 + 1}" for i in range(len(abundance_df))],
        'Timepoint': ['T1' if i % 2 == 0 else 'T2' for i in range(len(abundance_df))],
        'Age': np.random.normal(35, 10, len(abundance_df)).astype(int),
        'BMI': np.random.normal(24, 3, len(abundance_df))
    })
    
    print(f"✅ Loaded: {abundance_df.shape[0]} samples, {abundance_df.shape[1]} taxa")
    return abundance_df, metadata_df

def load_disease_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data for advanced analysis.
    
    Returns:
        Tuple of (abundance_dataframe, metadata_dataframe)
    """
    data_dir = "data/raw"
    abundance_file = os.path.join(data_dir, "disease_abundance.txt")
    metadata_file = os.path.join(data_dir, "disease_metadata.txt")
    
    if not os.path.exists(abundance_file) or not os.path.exists(metadata_file):
        print("📥 Data not found. Downloading sample data...")
        download_sample_data()
    
    print(f"📊 Loading disease analysis data...")
    
    # Read data
    abundance_df = pd.read_csv(abundance_file, sep='\t', index_col=0)
    metadata_df = pd.read_csv(metadata_file, sep='\t')
    
    print(f"✅ Loaded: {abundance_df.shape[0]} samples, {abundance_df.shape[1]} taxa")
    print(f"   Disease distribution: {metadata_df['Disease_Status'].value_counts().to_dict()}")
    
    return abundance_df, metadata_df

def validate_installation() -> bool:
    """
    Validate that all required packages are installed and working.
    
    Returns:
        True if all packages are working, False otherwise
    """
    print("🔍 Validating package installation...")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import entropy
        from sklearn.ensemble import RandomForestClassifier
        from statsmodels.stats.multitest import multipletests
        
        print("✅ Core packages: OK")
        
        # Test XGBoost separately with graceful handling
        xgb_available = False
        try:
            import xgboost as xgb
            xgb_available = True
            print("✅ XGBoost: OK")
        except Exception as e:
            print(f"⚠️  XGBoost: Not available ({str(e)[:50]}...)")
            print("   Note: XGBoost is optional. Random Forest will still work.")
        
        # Test basic functionality
        test_data = pd.DataFrame(np.random.rand(10, 5))
        shannon = entropy(test_data.iloc[0].values + 1e-6)
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        print("✅ Package functionality: OK")
        print(f"   Pandas version: {pd.__version__}")
        print(f"   NumPy version: {np.__version__}")
        print(f"   Scikit-learn version: {getattr(__import__('sklearn'), '__version__')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Package import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Main function to test data loading and package installation."""
    print("🧬 Microbiome Analysis - Data Loader Test")
    print("=" * 50)
    
    # Validate installation
    if not validate_installation():
        print("❌ Installation validation failed!")
        sys.exit(1)
    
    # Test data loading
    try:
        print("\n📊 Testing basic data loading...")
        abundance_basic, metadata_basic = load_basic_data()
        print(f"   Basic data shape: {abundance_basic.shape}")
        
        print("\n🏥 Testing disease data loading...")
        abundance_disease, metadata_disease = load_disease_data()
        print(f"   Disease data shape: {abundance_disease.shape}")
        
        print("\n✅ All tests passed! Ready for analysis.")
        print("\n🚀 Next steps:")
        print("   1. Run basic analysis: python basic_analysis/diversity_analysis.py")
        print("   2. Run advanced analysis: python advanced_analysis/ml_prediction.py")
        print("   3. Open Jupyter notebook: jupyter notebook notebooks/microbiome_analysis.ipynb")
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
