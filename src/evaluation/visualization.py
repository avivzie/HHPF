"""
Visualization for HHPF evaluation results.

Generates publication-quality figures for thesis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from src.utils import ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class HHPFVisualizer:
    """Create publication-quality visualizations for HHPF."""
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        # Figure settings
        self.fig_size = (10, 6)
        self.dpi = 300
    
    def plot_roc_curves(
        self,
        models_data: Dict[str, Dict],
        title: str = "ROC Curves Comparison",
        filename: str = "roc_curves"
    ):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_data: Dict mapping model name to metrics dict with 'roc_curve'
            title: Plot title
            filename: Output filename (without extension)
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        for model_name, metrics in models_data.items():
            roc_data = metrics['roc_curve']
            auroc = metrics['auroc']
            
            ax.plot(
                roc_data['fpr'],
                roc_data['tpr'],
                label=f'{model_name} (AUROC={auroc:.3f})',
                linewidth=2
            )
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUROC=0.500)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_arc(
        self,
        models_data: Dict[str, Dict],
        title: str = "Accuracy-Rejection Curve",
        filename: str = "arc_curve"
    ):
        """
        Plot Accuracy-Rejection Curves.
        
        Args:
            models_data: Dict mapping model name to metrics dict with 'arc'
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        for model_name, metrics in models_data.items():
            arc_data = metrics['arc']
            
            ax.plot(
                arc_data['rejection_rates'],
                arc_data['accuracies'],
                label=f'{model_name}',
                linewidth=2,
                marker='o',
                markersize=4
            )
        
        ax.set_xlabel('Rejection Rate', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_calibration(
        self,
        metrics: Dict,
        title: str = "Calibration Plot",
        filename: str = "calibration"
    ):
        """
        Plot Expected Calibration Error diagram.
        
        Args:
            metrics: Metrics dict with 'calibration' data
            title: Plot title
            filename: Output filename
        """
        calib_data = metrics['calibration']
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Bar chart of accuracies vs confidences
        import numpy as np
        bin_boundaries = np.array(calib_data['bin_boundaries'])
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_width = calib_data['bin_boundaries'][1] - calib_data['bin_boundaries'][0]
        
        # Plot bars for bin accuracies
        bars = ax.bar(
            bin_centers,
            calib_data['bin_accuracies'],
            width=bin_width * 0.8,
            alpha=0.7,
            label='Accuracy in bin',
            edgecolor='black'
        )
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        
        # Add bin counts as text
        for i, (center, acc, count) in enumerate(zip(bin_centers, calib_data['bin_accuracies'], calib_data['bin_counts'])):
            if not np.isnan(acc) and count > 0:
                ax.text(center, acc + 0.02, f'n={int(count)}', 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('True Accuracy', fontsize=12)
        ax.set_title(f'{title} (ECE={metrics["ece"]:.4f})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_k: int = 15,
        title: str = "Feature Importance",
        filename: str = "feature_importance"
    ):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_k: Number of top features to show
            title: Plot title
            filename: Output filename
        """
        # Select top k
        plot_df = importance_df.head(top_k).copy()
        plot_df = plot_df.sort_values('importance')  # Ascending for horizontal bar
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(
            range(len(plot_df)),
            plot_df['importance'],
            color='steelblue',
            edgecolor='black',
            alpha=0.8
        )
        
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_domain_comparison(
        self,
        domain_metrics: pd.DataFrame,
        metric_name: str = 'auroc',
        title: str = "Performance by Domain",
        filename: str = "domain_comparison"
    ):
        """
        Plot performance comparison across domains.
        
        Args:
            domain_metrics: DataFrame with domain and metric columns
            metric_name: Metric to plot
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        bars = ax.bar(
            range(len(domain_metrics)),
            domain_metrics[metric_name],
            color='coral',
            edgecolor='black',
            alpha=0.8
        )
        
        ax.set_xticks(range(len(domain_metrics)))
        ax.set_xticklabels(domain_metrics['domain'], fontsize=11)
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, domain_metrics[metric_name])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Feature Correlation Matrix",
        filename: str = "correlation_heatmap"
    ):
        """
        Plot correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_confusion_matrix(
        self,
        metrics: Dict,
        labels: List[str] = ['Faithful', 'Hallucination'],
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix"
    ):
        """
        Plot confusion matrix.
        
        Args:
            metrics: Metrics dict with confusion matrix values
            labels: Class labels
            title: Plot title
            filename: Output filename
        """
        cm = np.array([
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_feature_importance_by_domain(
        self,
        importance_pivot: pd.DataFrame,
        top_k: int = 10,
        title: str = "Feature Importance by Domain",
        filename: str = "feature_importance_domains"
    ):
        """
        Plot feature importance across domains.
        
        Args:
            importance_pivot: Pivot table with features as index, domains as columns
            top_k: Number of top features to show
            title: Plot title
            filename: Output filename
        """
        # Select top k features (by average importance)
        importance_pivot['avg'] = importance_pivot.mean(axis=1)
        top_features = importance_pivot.nlargest(top_k, 'avg').drop('avg', axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_features.plot(
            kind='barh',
            ax=ax,
            width=0.8
        )
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_semantic_vs_naive_comparison(
        self,
        ablation_results: pd.DataFrame,
        title: str = "Semantic Uncertainty vs Naive Confidence (RQ2)",
        filename: str = "semantic_vs_naive_comparison"
    ):
        """
        Plot comparison of semantic uncertainty vs naive baselines (RQ2).
        
        Args:
            ablation_results: DataFrame from ablation study
            title: Plot title
            filename: Output filename
        """
        # Filter relevant models
        models_to_plot = [
            'Baseline: Naive Confidence',
            'Semantic Uncertainty',
            'Semantic + Context',
            'Full Model'
        ]
        
        plot_df = ablation_results[ablation_results['model'].isin(models_to_plot)].copy()
        
        if len(plot_df) == 0:
            logger.warning("No data for semantic vs naive comparison")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = range(len(plot_df))
        bars = ax.bar(
            x_pos,
            plot_df['auroc'],
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            edgecolor='black',
            alpha=0.8
        )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_df['model'], fontsize=11, rotation=15, ha='right')
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0.5, 1.0)  # AUROC range
        
        # Add value labels and improvement annotations
        for i, (bar, row) in enumerate(zip(bars, plot_df.itertuples())):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{row.auroc:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
            
            # Show improvement over naive
            if i > 0:
                naive_auroc = plot_df.iloc[0]['auroc']
                improvement = row.auroc - naive_auroc
                if improvement > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height - 0.05,
                        f'+{improvement:.3f}',
                        ha='center',
                        va='top',
                        fontsize=9,
                        color='green',
                        fontweight='bold'
                    )
        
        # Add baseline line
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.500)')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_domain_specific_auroc(
        self,
        domain_metrics: Dict[str, Dict],
        title: str = "Model Performance by Domain (RQ3)",
        filename: str = "domain_specific_auroc"
    ):
        """
        Plot AUROC comparison across domains (RQ3).
        
        Args:
            domain_metrics: Dict mapping domain to metrics
            title: Plot title
            filename: Output filename
        """
        domains = list(domain_metrics.keys())
        aurocs = [domain_metrics[d]['metrics']['auroc'] for d in domains]
        n_samples = [domain_metrics[d]['n_samples'] for d in domains]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(
            range(len(domains)),
            aurocs,
            color='steelblue',
            edgecolor='black',
            alpha=0.8
        )
        
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, fontsize=11)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0.5, 1.0)
        
        # Add value labels and sample counts
        for i, (bar, auroc, n) in enumerate(zip(bars, aurocs, n_samples)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{auroc:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.52,
                f'n={n}',
                ha='center',
                va='bottom',
                fontsize=8,
                style='italic'
            )
        
        # Baseline
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')
        ax.legend()
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def plot_domain_feature_importance_heatmap(
        self,
        domain_models: Dict[str, Dict],
        top_k: int = 15,
        title: str = "Feature Importance Across Domains (RQ3)",
        filename: str = "domain_feature_heatmap"
    ):
        """
        Plot heatmap of feature importance across domains (RQ3).
        
        Args:
            domain_models: Dict mapping domain to model data
            top_k: Number of top features to show
            title: Plot title
            filename: Output filename
        """
        # Collect importance data
        importance_data = []
        
        for domain, model_data in domain_models.items():
            importance_df = model_data['feature_importance']
            
            for _, row in importance_df.iterrows():
                importance_data.append({
                    'domain': domain,
                    'feature': row['feature'],
                    'importance': row['importance']
                })
        
        importance_df = pd.DataFrame(importance_data)
        
        # Pivot to create heatmap data
        importance_pivot = importance_df.pivot(
            index='feature',
            columns='domain',
            values='importance'
        )
        
        # Select top k features (by average importance)
        importance_pivot['avg'] = importance_pivot.mean(axis=1)
        top_features = importance_pivot.nlargest(top_k, 'avg').drop('avg', axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            top_features,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            linewidths=0.5,
            cbar_kws={'label': 'Importance'},
            ax=ax
        )
        
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()
    
    def _save_figure(self, fig, filename: str):
        """Save figure in multiple formats."""
        # Save as PDF (vector graphics for LaTeX)
        pdf_path = self.output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, format='pdf', dpi=self.dpi, bbox_inches='tight')
        
        # Save as PNG (high resolution for presentations)
        png_path = self.output_dir / f"{filename}.png"
        fig.savefig(png_path, format='png', dpi=self.dpi, bbox_inches='tight')
        
        logger.info(f"✓ Saved figure: {filename}")


def main():
    """Main entry point for visualization."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Generate HHPF visualizations")
    parser.add_argument(
        '--metrics',
        type=str,
        required=True,
        help='Path to metrics JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/figures',
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    # Load metrics
    with open(args.metrics, 'r') as f:
        results = json.load(f)
    
    # Initialize visualizer
    viz = HHPFVisualizer(output_dir=args.output_dir)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # ROC curve
    models_data = {
        'Full Model': results['test']
    }
    viz.plot_roc_curves(models_data, filename='roc_curve_test')
    
    # ARC
    viz.plot_arc(models_data, filename='arc_test')
    
    # Calibration
    viz.plot_calibration(results['test'], filename='calibration_test')
    
    # Confusion matrix
    viz.plot_confusion_matrix(results['test'], filename='confusion_matrix_test')
    
    logger.info(f"\n✓ All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
