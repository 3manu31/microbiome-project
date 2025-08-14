# Microbiome Data Analysis Project

A comprehensive Python-based analysis workflow for gut microbiota data, featuring both basic exploratory analysis and advanced machine learning approaches for disease prediction.

## Project Overview

### Part 1: Basic Analysis
- **Goal**: Quick exploration of gut microbiota diversity
- **Dataset**: Human Microbiome Project (HMP) gut samples
- **Skills**: Data import, diversity metrics, visualization
- **Outputs**: Shannon diversity calculations, taxa abundance plots

### Part 2: Advanced Analysis  
- **Goal**: Disease prediction from gut microbiota using ML
- **Dataset**: Public metagenomic data with health metadata
- **Skills**: ML classification, statistical analysis, feature selection
- **Outputs**: Trained models, statistical comparisons, publication-ready figures

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic analysis**:
   ```bash
   python basic_analysis/diversity_analysis.py
   ```

3. **Run advanced analysis**:
   ```bash
   python advanced_analysis/ml_prediction.py
   ```

## Project Structure

```
microbiome-project/
├── basic_analysis/           # Part 1: Basic diversity analysis
│   ├── diversity_analysis.py
│   ├── data_loader.py
│   └── plots/
├── advanced_analysis/        # Part 2: ML-based prediction
│   ├── ml_prediction.py
│   ├── preprocessing.py
│   ├── statistical_analysis.py
│   └── results/
├── data/                    # Dataset storage
│   ├── raw/
│   └── processed/
├── notebooks/               # Jupyter notebooks
├── utils/                   # Shared utilities
└── requirements.txt
```

## Datasets

### Basic Analysis
- **Source**: Human Microbiome Project (HMP)
- **Type**: Processed abundance tables
- **Samples**: Gut microbiota from healthy individuals
- **Format**: Tab-separated values

### Advanced Analysis
- **Source**: MGnify or similar public repositories
- **Type**: Abundance + metadata (disease status)
- **Samples**: IBS vs. healthy controls
- **Format**: OTU/ASV tables with clinical metadata

## Dependencies

- **Core**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **ML**: scikit-learn, xgboost, imbalanced-learn
- **Bioinformatics**: biopython, skbio
- **Statistics**: statsmodels, pingouin

## Expected Outputs

### Basic Analysis
- Shannon diversity index calculations
- Top taxa relative abundance bar plots
- Alpha diversity comparison plots
- Summary statistics tables

### Advanced Analysis
- Trained Random Forest/XGBoost models
- Model performance metrics (accuracy, precision, recall)
- Feature importance plots
- Statistical comparison of taxa between groups
- Publication-quality figures (300 DPI PNG)

## Analysis Workflow

1. **Data Loading**: Import and validate microbiome datasets
2. **Quality Control**: Filter low-abundance taxa, handle missing data
3. **Normalization**: Apply appropriate transformations (CLR, log, relative abundance)
4. **Diversity Analysis**: Calculate alpha/beta diversity metrics
5. **Statistical Testing**: Compare groups using appropriate tests
6. **Machine Learning**: Train classifiers for disease prediction
7. **Visualization**: Generate publication-ready plots

## References

- Human Microbiome Project: https://www.hmpdacc.org/
- MGnify Database: https://www.ebi.ac.uk/metagenomics/
- scikit-bio Documentation: http://scikit-bio.org/

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

## Attribution & Citation

If you use this code or workflow in your research, publication, or project, please credit as follows:

**Suggested citation:**

**Suggested citation:**

Emmanuel Gialitakis, microbiome-project, GitHub, 2025. Available at: [https://github.com/3manu31/microbiome-project](https://github.com/3manu31/microbiome-project)


**Attribution requirement (MIT License):**

- Please retain the copyright notice and license text in any copies or substantial portions of the code.
- In publications, you may cite the repository and/or mention the author in the methods, acknowledgments, or bibliography.

Example for publications:
Example for publications:
> "Analysis performed using the microbiome-project (Emmanuel Gialitakis, GitHub, 2025, [https://github.com/3manu31/microbiome-project](https://github.com/3manu31/microbiome-project))."
