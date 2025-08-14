"""
Basic Microbiome Analysis - Part 1
==================================

This script performs basic diversity analysis on Human Microbiome Project gut data:
1. Downloads HMP gut microbiota abundance data
2. Calculates Shannon diversity index
3. Visualizes top taxa abundance
4. Generates summary statistics

Author: Microbiome Analysis Project
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BasicMicrobiomeAnalysis:
    """Class for basic microbiome diversity analysis."""
    
    def __init__(self, data_dir="../data"):
        """Initialize with data directory path."""
        self.data_dir = data_dir
        self.abundance_data = None
        self.metadata = None
        self.diversity_results = None
        
        # Create output directory
        self.output_dir = "plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_sample_data(self):
        """
        Load or create sample HMP-style gut microbiota data.
        In a real scenario, this would download from HMP portal.
        """
        print("Loading sample gut microbiota abundance data...")
        
        # For demo purposes, create realistic sample data
        # In real analysis, this would be: self.load_hmp_data()
        samples = [f"Sample_{i:03d}" for i in range(1, 51)]  # 50 samples
        taxa = [
            "Bacteroides", "Faecalibacterium", "Prevotella", "Ruminococcus",
            "Eubacterium", "Clostridium", "Bifidobacterium", "Lactobacillus",
            "Enterobacteriaceae", "Akkermansia", "Roseburia", "Coprococcus",
            "Blautia", "Dialister", "Parabacteroides", "Alistipes",
            "Oscillospira", "Sutterella", "Collinsella", "Dorea"
        ]
        
        # Generate realistic abundance data (Dirichlet distribution)
        np.random.seed(42)
        alpha = np.random.gamma(2, 2, len(taxa))  # Different abundances per taxa
        abundance_matrix = np.random.dirichlet(alpha, len(samples))
        
        # Create DataFrame
        self.abundance_data = pd.DataFrame(
            abundance_matrix,
            index=samples,
            columns=taxa
        )
        
        # Create simple metadata
        self.metadata = pd.DataFrame({
            'SampleID': samples,
            'Subject': [f"Subject_{i//2 + 1}" for i in range(len(samples))],
            'Timepoint': ['T1' if i % 2 == 0 else 'T2' for i in range(len(samples))],
            'Age': np.random.normal(35, 10, len(samples)).astype(int),
            'BMI': np.random.normal(24, 3, len(samples))
        })
        
        print(f"Loaded data: {self.abundance_data.shape[0]} samples, {self.abundance_data.shape[1]} taxa")
        return self.abundance_data, self.metadata
    
    def calculate_shannon_diversity(self):
        """Calculate Shannon diversity index for each sample."""
        print("Calculating Shannon diversity indices...")
        
        shannon_values = []
        for sample in self.abundance_data.index:
            abundances = self.abundance_data.loc[sample].values
            # Remove zero abundances
            abundances = abundances[abundances > 0]
            # Calculate Shannon index
            shannon = entropy(abundances, base=np.e)
            shannon_values.append(shannon)
        
        self.diversity_results = pd.DataFrame({
            'SampleID': self.abundance_data.index,
            'Shannon_Diversity': shannon_values
        })
        
        # Merge with metadata
        self.diversity_results = self.diversity_results.merge(
            self.metadata, on='SampleID', how='left'
        )
        
        print(f"Shannon diversity calculated for {len(shannon_values)} samples")
        print(f"Mean Shannon diversity: {np.mean(shannon_values):.3f} ± {np.std(shannon_values):.3f}")
        
        return self.diversity_results
    
    def plot_diversity_distribution(self):
        """Plot Shannon diversity distribution."""
        plt.figure(figsize=(10, 6))
        
        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.diversity_results, x='Shannon_Diversity', 
                    kde=True, bins=15, alpha=0.7)
        plt.title('Shannon Diversity Distribution')
        plt.xlabel('Shannon Diversity Index')
        plt.ylabel('Frequency')
        
        # Box plot by timepoint
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.diversity_results, x='Timepoint', y='Shannon_Diversity')
        plt.title('Shannon Diversity by Timepoint')
        plt.ylabel('Shannon Diversity Index')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/shannon_diversity_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_top_taxa_abundance(self, top_n=10):
        """Plot relative abundance of top taxa."""
        print(f"Plotting top {top_n} taxa abundance...")
        
        # Calculate mean abundance for each taxa
        mean_abundance = self.abundance_data.mean().sort_values(ascending=False)
        top_taxa = mean_abundance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Bar plot of mean abundance
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(top_taxa)), top_taxa.values)
        plt.xticks(range(len(top_taxa)), top_taxa.index, rotation=45, ha='right')
        plt.ylabel('Mean Relative Abundance')
        plt.title(f'Top {top_n} Taxa by Mean Relative Abundance')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Heatmap of top taxa across samples
        plt.subplot(2, 1, 2)
        top_taxa_data = self.abundance_data[top_taxa.index]
        sns.heatmap(top_taxa_data.T, cmap='viridis', cbar_kws={'label': 'Relative Abundance'})
        plt.title(f'Top {top_n} Taxa Abundance Across Samples')
        plt.xlabel('Samples')
        plt.ylabel('Taxa')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/top_taxa_abundance.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_taxa
    
    def plot_taxa_correlation(self, top_n=10):
        """Plot correlation matrix of top taxa."""
        print("Generating taxa correlation matrix...")
        
        # Get top taxa
        mean_abundance = self.abundance_data.mean().sort_values(ascending=False)
        top_taxa = mean_abundance.head(top_n).index
        
        # Calculate correlation
        correlation_matrix = self.abundance_data[top_taxa].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title(f'Taxa Correlation Matrix (Top {top_n})')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/taxa_correlation.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_stats(self):
        """Generate summary statistics."""
        print("\n" + "="*50)
        print("BASIC MICROBIOME ANALYSIS SUMMARY")
        print("="*50)
        
        # Sample statistics
        print(f"Total samples: {self.abundance_data.shape[0]}")
        print(f"Total taxa: {self.abundance_data.shape[1]}")
        
        # Diversity statistics
        diversity_stats = self.diversity_results['Shannon_Diversity'].describe()
        print(f"\nShannon Diversity Statistics:")
        print(f"Mean: {diversity_stats['mean']:.3f}")
        print(f"Std:  {diversity_stats['std']:.3f}")
        print(f"Min:  {diversity_stats['min']:.3f}")
        print(f"Max:  {diversity_stats['max']:.3f}")
        
        # Taxa abundance statistics
        print(f"\nTaxa Abundance Statistics:")
        print(f"Most abundant taxa: {self.abundance_data.mean().idxmax()}")
        print(f"Highest mean abundance: {self.abundance_data.mean().max():.4f}")
        print(f"Lowest mean abundance: {self.abundance_data.mean().min():.4f}")
        
        # Save summary to file
        summary_file = f"{self.output_dir}/analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("BASIC MICROBIOME ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis Date: August 2025\n")
            f.write(f"Total samples: {self.abundance_data.shape[0]}\n")
            f.write(f"Total taxa: {self.abundance_data.shape[1]}\n\n")
            f.write("Shannon Diversity Statistics:\n")
            f.write(f"Mean: {diversity_stats['mean']:.3f}\n")
            f.write(f"Std:  {diversity_stats['std']:.3f}\n")
            f.write(f"Min:  {diversity_stats['min']:.3f}\n")
            f.write(f"Max:  {diversity_stats['max']:.3f}\n\n")
            f.write("Taxa Abundance Statistics:\n")
            f.write(f"Most abundant taxa: {self.abundance_data.mean().idxmax()}\n")
            f.write(f"Highest mean abundance: {self.abundance_data.mean().max():.4f}\n")
            f.write(f"Lowest mean abundance: {self.abundance_data.mean().min():.4f}\n")
        
        print(f"\nSummary saved to: {summary_file}")
        print("="*50)
    
    def run_complete_analysis(self):
        """Run the complete basic analysis workflow."""
        print("Starting Basic Microbiome Analysis...")
        print("="*50)
        
        # Load data
        self.load_sample_data()
        
        # Calculate diversity
        self.calculate_shannon_diversity()
        
        # Generate visualizations
        self.plot_diversity_distribution()
        self.plot_top_taxa_abundance(top_n=10)
        self.plot_taxa_correlation(top_n=8)
        
        # Generate summary
        self.generate_summary_stats()
        
        print(f"\nAnalysis complete! Results saved in '{self.output_dir}/' directory")
        return self.abundance_data, self.diversity_results


def main():
    """Main function to run basic microbiome analysis."""
    # Initialize analysis
    analyzer = BasicMicrobiomeAnalysis()
    
    # Run complete analysis
    abundance_data, diversity_results = analyzer.run_complete_analysis()
    
    return abundance_data, diversity_results


if __name__ == "__main__":
    main()
