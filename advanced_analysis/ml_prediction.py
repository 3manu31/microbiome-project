"""
Advanced Microbiome Analysis - Part 2
=====================================

This script performs advanced ML-based analysis for disease prediction:
1. Loads abundance data with health metadata
2. Preprocesses and normalizes data
3. Performs feature selection
4. Trains Random Forest/XGBoost classifiers
5. Conducts statistical analysis of taxa differences
6. Generates publication-quality visualizations

Author: Microbiome Analysis Project
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fall back gracefully if not available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception as e:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available, using only Random Forest")
    print(f"   Error: {str(e)[:100]}...")
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

class AdvancedMicrobiomeAnalysis:
    """Class for advanced ML-based microbiome analysis."""
    
    def __init__(self, data_dir="../data"):
        """Initialize with data directory path."""
        self.data_dir = data_dir
        self.abundance_data = None
        self.metadata = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.feature_importance = {}
        self.statistical_results = None
        
        # Create output directory
        self.output_dir = "results"
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_disease_data(self):
        """
        Load or create sample microbiome data with disease labels.
        In real analysis, this would load from MGnify or similar databases.
        """
        print("Loading microbiome data with disease metadata...")
        
        # Generate realistic disease vs healthy microbiome data
        np.random.seed(42)
        n_samples = 200
        n_taxa = 50
        
        # Create taxa names (mix of real and simulated)
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
        # 60% healthy, 40% IBS
        disease_labels = ["Healthy"] * int(n_samples * 0.6) + ["IBS"] * int(n_samples * 0.4)
        np.random.shuffle(disease_labels)
        
        # Generate abundance data with disease-specific patterns
        abundance_matrix = []
        for i, label in enumerate(disease_labels):
            if label == "Healthy":
                # Healthy microbiome: higher diversity, more beneficial bacteria
                alpha = np.random.gamma(2, 1, n_taxa)
                alpha[1] *= 2  # More Faecalibacterium (beneficial)
                alpha[9] *= 1.5  # More Akkermansia (beneficial)
            else:
                # IBS microbiome: lower diversity, dysbiosis
                alpha = np.random.gamma(1.5, 0.8, n_taxa)
                alpha[0] *= 1.5  # More Bacteroides (associated with IBS)
                alpha[8] *= 2  # More Enterobacteriaceae (pathogenic)
                alpha[1] *= 0.5  # Less Faecalibacterium
                alpha[9] *= 0.7  # Less Akkermansia
            
            abundances = np.random.dirichlet(alpha)
            abundance_matrix.append(abundances)
        
        # Create abundance DataFrame
        self.abundance_data = pd.DataFrame(
            abundance_matrix,
            index=sample_ids,
            columns=taxa_names
        )
        
        # Create metadata
        self.metadata = pd.DataFrame({
            'SampleID': sample_ids,
            'Disease_Status': disease_labels,
            'Age': np.random.normal(45, 15, n_samples).astype(int),
            'BMI': np.random.normal(25, 4, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Antibiotics_Recent': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        })
        
        print(f"Loaded data: {len(sample_ids)} samples, {n_taxa} taxa")
        print(f"Disease distribution: {pd.Series(disease_labels).value_counts()}")
        
        return self.abundance_data, self.metadata
    
    def preprocess_data(self, filter_threshold=0.001, clr_transform=True):
        """
        Preprocess microbiome data:
        - Filter low-abundance taxa
        - Apply CLR transformation
        - Handle zeros
        """
        print("Preprocessing microbiome data...")
        
        # Filter low-abundance taxa
        mean_abundance = self.abundance_data.mean()
        high_abundance_taxa = mean_abundance[mean_abundance >= filter_threshold].index
        filtered_data = self.abundance_data[high_abundance_taxa]
        print(f"Filtered taxa: {len(high_abundance_taxa)} / {len(mean_abundance)} retained")
        
        if clr_transform:
            # Apply Centered Log-Ratio (CLR) transformation
            # Add pseudocount to handle zeros
            pseudocount = 1e-6
            data_with_pseudo = filtered_data + pseudocount
            
            # CLR transformation
            geometric_mean = np.exp(np.log(data_with_pseudo).mean(axis=1))
            clr_data = np.log(data_with_pseudo.div(geometric_mean, axis=0))
            
            self.processed_data = clr_data
            print("Applied CLR transformation")
        else:
            # Use relative abundance
            self.processed_data = filtered_data
            print("Using relative abundance data")
        
        return self.processed_data
    
    def split_data(self, test_size=0.3, random_state=42):
        """Split data into training and testing sets."""
        print("Splitting data into train/test sets...")
        
        X = self.processed_data
        y = self.metadata['Disease_Status']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Training labels: {self.y_train.value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def feature_selection(self, method='rfe', k=20):
        """Perform feature selection to identify important taxa."""
        print(f"Performing feature selection using {method}...")
        
        if method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(score_func=f_classif, k=k)
            self.X_train_selected = selector.fit_transform(self.X_train, self.y_train)
            self.X_test_selected = selector.transform(self.X_test)
            selected_features = self.X_train.columns[selector.get_support()]
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=rf_temp, n_features_to_select=k)
            self.X_train_selected = selector.fit_transform(self.X_train, self.y_train)
            self.X_test_selected = selector.transform(self.X_test)
            selected_features = self.X_train.columns[selector.support_]
        
        self.selected_features = selected_features
        print(f"Selected {len(selected_features)} features")
        print("Top selected taxa:", list(selected_features[:5]))
        
        return selected_features
    
    def train_models(self):
        """Train Random Forest and optionally XGBoost classifiers."""
        print("Training machine learning models...")
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf_model.fit(self.X_train_selected, self.y_train)
        self.models['RandomForest'] = rf_model
        
        # XGBoost (if available)
        if XGB_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(self.X_train_selected, self.y_train)
            self.models['XGBoost'] = xgb_model
        else:
            print("   XGBoost not available, training only Random Forest")
        
        print(f"Models trained successfully: {list(self.models.keys())}")
        return self.models
    
    def evaluate_models(self):
        """Evaluate model performance using cross-validation and test set."""
        print("Evaluating model performance...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        
        for name, model in self.models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train_selected, self.y_train, 
                                      cv=cv, scoring='accuracy')
            
            # Test set predictions
            y_pred = model.predict(self.X_test_selected)
            y_pred_proba = model.predict_proba(self.X_test_selected)[:, 1]
            
            # Calculate metrics
            test_accuracy = (y_pred == self.y_test).mean()
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"\n{name} Results:")
            print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"Test Accuracy: {test_accuracy:.3f}")
            print(f"AUC Score: {auc_score:.3f}")
        
        self.model_results = results
        return results
    
    def statistical_analysis(self):
        """Perform statistical analysis of taxa differences between groups."""
        print("Performing statistical analysis of taxa differences...")
        
        # Align metadata with processed data indices
        metadata_aligned = self.metadata.set_index('SampleID').loc[self.processed_data.index]
        
        # Separate groups
        healthy_data = self.processed_data[metadata_aligned['Disease_Status'] == 'Healthy']
        ibs_data = self.processed_data[metadata_aligned['Disease_Status'] == 'IBS']
        
        statistical_results = []
        
        for taxa in self.processed_data.columns:
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = mannwhitneyu(
                healthy_data[taxa], ibs_data[taxa], alternative='two-sided'
            )
            
            # Effect size (fold change)
            healthy_mean = healthy_data[taxa].mean()
            ibs_mean = ibs_data[taxa].mean()
            log_fold_change = np.log2((ibs_mean + 1e-6) / (healthy_mean + 1e-6))
            
            statistical_results.append({
                'Taxa': taxa,
                'Healthy_Mean': healthy_mean,
                'IBS_Mean': ibs_mean,
                'Log_Fold_Change': log_fold_change,
                'P_Value': p_value,
                'Statistic': statistic
            })
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(statistical_results)
        
        # FDR correction
        rejected, p_adjusted, _, _ = multipletests(
            stats_df['P_Value'], alpha=0.05, method='fdr_bh'
        )
        stats_df['P_Adjusted'] = p_adjusted
        stats_df['Significant'] = rejected
        
        # Sort by significance
        stats_df = stats_df.sort_values('P_Adjusted')
        
        self.statistical_results = stats_df
        
        print(f"Significant taxa (FDR < 0.05): {stats_df['Significant'].sum()}")
        print("Top 5 significant taxa:")
        print(stats_df.head()[['Taxa', 'Log_Fold_Change', 'P_Adjusted']])
        
        return stats_df
    
    def plot_model_performance(self):
        """Plot model performance metrics."""
        print("Generating model performance plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy comparison
        model_names = list(self.model_results.keys())
        cv_accuracies = [self.model_results[name]['cv_accuracy_mean'] for name in model_names]
        test_accuracies = [self.model_results[name]['test_accuracy'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, cv_accuracies, width, label='CV Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. ROC Curves
        for name, results in self.model_results.items():
            y_test_binary = (self.y_test == 'IBS').astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, results['y_pred_proba'])
            auc = results['auc_score']
            axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Confusion Matrix for best model
        best_model_name = max(self.model_results.keys(), 
                             key=lambda x: self.model_results[x]['test_accuracy'])
        y_pred_best = self.model_results[best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred_best)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        
        # 4. Feature Importance
        if best_model_name == 'RandomForest':
            importances = self.models[best_model_name].feature_importances_
        else:  # XGBoost
            importances = self.models[best_model_name].feature_importances_
        
        # Get top 10 features
        feature_importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)
        
        axes[1, 1].barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
        axes[1, 1].set_yticks(range(len(feature_importance_df)))
        axes[1, 1].set_yticklabels(feature_importance_df['Feature'])
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title(f'Top 10 Feature Importance - {best_model_name}')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_performance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_statistical_results(self):
        """Plot statistical analysis results."""
        print("Generating statistical analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Volcano plot
        stats_df = self.statistical_results.copy()
        stats_df['-log10_p'] = -np.log10(stats_df['P_Adjusted'] + 1e-10)
        
        # Color by significance
        colors = ['red' if sig else 'gray' for sig in stats_df['Significant']]
        axes[0, 0].scatter(stats_df['Log_Fold_Change'], stats_df['-log10_p'], 
                          c=colors, alpha=0.6)
        axes[0, 0].axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].set_xlabel('Log2 Fold Change (IBS vs Healthy)')
        axes[0, 0].set_ylabel('-log10(Adjusted P-value)')
        axes[0, 0].set_title('Volcano Plot: Taxa Differential Abundance')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Top significant taxa
        top_significant = stats_df[stats_df['Significant']].head(10)
        if len(top_significant) > 0:
            axes[0, 1].barh(range(len(top_significant)), top_significant['Log_Fold_Change'])
            axes[0, 1].set_yticks(range(len(top_significant)))
            axes[0, 1].set_yticklabels(top_significant['Taxa'])
            axes[0, 1].set_xlabel('Log2 Fold Change')
            axes[0, 1].set_title('Top Differentially Abundant Taxa')
            axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. P-value distribution
        axes[1, 0].hist(stats_df['P_Value'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('P-value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('P-value Distribution')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Group comparison for top taxa
        if len(top_significant) > 0:
            top_taxa = top_significant.iloc[0]['Taxa']
            # Use aligned metadata
            metadata_aligned = self.metadata.set_index('SampleID').loc[self.processed_data.index]
            healthy_vals = self.processed_data[metadata_aligned['Disease_Status'] == 'Healthy'][top_taxa]
            ibs_vals = self.processed_data[metadata_aligned['Disease_Status'] == 'IBS'][top_taxa]
            
            data_for_plot = pd.DataFrame({
                'Value': list(healthy_vals) + list(ibs_vals),
                'Group': ['Healthy'] * len(healthy_vals) + ['IBS'] * len(ibs_vals)
            })
            
            sns.boxplot(data=data_for_plot, x='Group', y='Value', ax=axes[1, 1])
            sns.swarmplot(data=data_for_plot, x='Group', y='Value', ax=axes[1, 1], 
                         color='black', alpha=0.5, size=3)
            axes[1, 1].set_title(f'Abundance Comparison: {top_taxa}')
            axes[1, 1].set_ylabel('CLR-transformed Abundance')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/statistical_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*60)
        print("ADVANCED MICROBIOME ANALYSIS REPORT")
        print("="*60)
        
        # Dataset summary
        print(f"Dataset: {self.abundance_data.shape[0]} samples, {self.abundance_data.shape[1]} taxa")
        print(f"Disease distribution: {self.metadata['Disease_Status'].value_counts().to_dict()}")
        
        # Feature selection summary
        print(f"\nFeature Selection: {len(self.selected_features)} taxa selected")
        
        # Model performance summary
        print(f"\nModel Performance:")
        for name, results in self.model_results.items():
            print(f"{name}:")
            print(f"  CV Accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}")
            print(f"  Test Accuracy: {results['test_accuracy']:.3f}")
            print(f"  AUC Score: {results['auc_score']:.3f}")
        
        # Statistical analysis summary
        significant_taxa = self.statistical_results['Significant'].sum()
        print(f"\nStatistical Analysis:")
        print(f"Significant taxa (FDR < 0.05): {significant_taxa}")
        
        if significant_taxa > 0:
            top_taxa = self.statistical_results[self.statistical_results['Significant']].iloc[0]
            print(f"Most significant taxa: {top_taxa['Taxa']}")
            print(f"Log2 fold change: {top_taxa['Log_Fold_Change']:.3f}")
            print(f"Adjusted p-value: {top_taxa['P_Adjusted']:.2e}")
        
        # Save report
        report_file = f"{self.output_dir}/analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write("ADVANCED MICROBIOME ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Analysis Date: August 2025\n")
            f.write(f"Dataset: {self.abundance_data.shape[0]} samples, {self.abundance_data.shape[1]} taxa\n")
            f.write(f"Disease distribution: {self.metadata['Disease_Status'].value_counts().to_dict()}\n\n")
            f.write(f"Feature Selection: {len(self.selected_features)} taxa selected\n\n")
            f.write("Model Performance:\n")
            for name, results in self.model_results.items():
                f.write(f"{name}:\n")
                f.write(f"  CV Accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}\n")
                f.write(f"  Test Accuracy: {results['test_accuracy']:.3f}\n")
                f.write(f"  AUC Score: {results['auc_score']:.3f}\n")
            f.write(f"\nStatistical Analysis:\n")
            f.write(f"Significant taxa (FDR < 0.05): {significant_taxa}\n")
            if significant_taxa > 0:
                f.write(f"Most significant taxa: {top_taxa['Taxa']}\n")
                f.write(f"Log2 fold change: {top_taxa['Log_Fold_Change']:.3f}\n")
                f.write(f"Adjusted p-value: {top_taxa['P_Adjusted']:.2e}\n")
        
        print(f"\nReport saved to: {report_file}")
        print("="*60)
    
    def run_complete_analysis(self):
        """Run the complete advanced analysis workflow."""
        print("Starting Advanced Microbiome Analysis...")
        print("="*60)
        
        # Load data
        self.load_disease_data()
        
        # Preprocess
        self.preprocess_data()
        
        # Split data
        self.split_data()
        
        # Feature selection
        self.feature_selection(method='rfe', k=20)
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Statistical analysis
        self.statistical_analysis()
        
        # Generate visualizations
        self.plot_model_performance()
        self.plot_statistical_results()
        
        # Generate report
        self.generate_report()
        
        print(f"\nAdvanced analysis complete! Results saved in '{self.output_dir}/' directory")
        return self.models, self.statistical_results


def main():
    """Main function to run advanced microbiome analysis."""
    # Initialize analysis
    analyzer = AdvancedMicrobiomeAnalysis()
    
    # Run complete analysis
    models, statistical_results = analyzer.run_complete_analysis()
    
    return analyzer


if __name__ == "__main__":
    main()
