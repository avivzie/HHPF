"""
Generate individual domain visualization figures using trained models.

Uses the newly trained consistent models and saved metrics to generate:
- ROC Curve (with AUROC)
- ARC (Accuracy-Rejection Curve)
- Calibration Plot (with ECE)
- Confusion Matrix
- Feature Importance (top 15)

Output: outputs/figures/{domain}/
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_roc_curve(metrics, domain, output_dir):
    """Plot ROC curve with AUROC."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fpr = metrics['roc_curve']['fpr']
    tpr = metrics['roc_curve']['tpr']
    auroc = metrics['auroc']
    
    ax.plot(fpr, tpr, label=f'Full Model (AUROC={auroc:.3f})', 
            linewidth=2, color='#FF1493')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUROC=0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'roc_curve_{domain}.{ext}'
        plt.savefig(path, bbox_inches='tight', dpi=300)
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_arc(metrics, domain, output_dir):
    """Plot Accuracy-Rejection Curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rejection_rates = metrics['arc']['rejection_rates']
    accuracies = metrics['arc']['accuracies']
    
    ax.plot(rejection_rates, accuracies, label='Full Model',
            linewidth=2, marker='o', markersize=4, color='#1f77b4')
    
    ax.set_xlabel('Rejection Rate', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy-Rejection Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'arc_{domain}.{ext}'
        plt.savefig(path, bbox_inches='tight', dpi=300)
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_calibration(metrics, domain, output_dir):
    """Plot calibration diagram."""
    calib_data = metrics['calibration']
    ece = metrics['ece']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bin_boundaries = np.array(calib_data['bin_boundaries'])
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_width = bin_boundaries[1] - bin_boundaries[0]
    
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
    for i, (center, acc, count) in enumerate(zip(bin_centers, 
                                                   calib_data['bin_accuracies'],
                                                   calib_data['bin_counts'])):
        if not np.isnan(acc) and count > 0:
            ax.text(center, acc + 0.02, f'n={int(count)}', 
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Accuracy', fontsize=12)
    ax.set_title(f'Calibration Plot (ECE={ece:.4f})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'calibration_{domain}.{ext}'
        plt.savefig(path, bbox_inches='tight', dpi=300)
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_confusion_matrix(metrics, domain, output_dir):
    """Plot confusion matrix."""
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
        xticklabels=['Faithful', 'Hallucination'],
        yticklabels=['Faithful', 'Hallucination'],
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'confusion_matrix_{domain}.{ext}'
        plt.savefig(path, bbox_inches='tight', dpi=300)
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_feature_importance(importance_df, domain, output_dir):
    """Plot feature importance."""
    # Select top 15
    plot_df = importance_df.head(15).copy()
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
    ax.set_title(f'Feature Importance - {domain.capitalize()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'feature_importance_{domain}.{ext}'
        plt.savefig(path, bbox_inches='tight', dpi=300)
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def generate_domain_figures(domain, metrics_path, importance_path, output_dir):
    """Generate all figures for a domain."""
    logger.info("="*80)
    logger.info(f"GENERATING FIGURES: {domain.upper()}")
    logger.info("="*80)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        data = json.load(f)
        metrics = data['test']
    
    logger.info(f"\nMetrics loaded:")
    logger.info(f"  AUROC: {metrics['auroc']:.4f}")
    logger.info(f"  ECE: {metrics['ece']:.4f}")
    
    # Load feature importance
    importance_df = pd.read_csv(importance_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all figures
    logger.info(f"\nGenerating figures...")
    plot_roc_curve(metrics, domain, output_dir)
    plot_arc(metrics, domain, output_dir)
    plot_calibration(metrics, domain, output_dir)
    plot_confusion_matrix(metrics, domain, output_dir)
    plot_feature_importance(importance_df, domain, output_dir)
    
    logger.info(f"\n✅ {domain.upper()} FIGURES COMPLETE")
    logger.info(f"All figures saved to {output_dir}/\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate individual domain visualization figures"
    )
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['math', 'is_agents', 'psychology', 'medicine', 'finance'],
        help='Domain to generate figures for'
    )
    parser.add_argument(
        '--metrics-dir',
        type=str,
        default='outputs/results',
        help='Directory containing metrics JSON files'
    )
    parser.add_argument(
        '--ablation-dir',
        type=str,
        default='outputs/ablation',
        help='Directory containing feature importance CSVs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/figures/{domain}/)'
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f'outputs/figures/{args.domain}'
    
    # Construct file paths
    metrics_path = Path(args.metrics_dir) / f"metrics_{args.domain}.json"
    importance_path = Path(args.ablation_dir) / f"{args.domain}_feature_importance.csv"
    
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return 1
    
    if not importance_path.exists():
        logger.error(f"Feature importance file not found: {importance_path}")
        return 1
    
    # Generate figures
    try:
        generate_domain_figures(
            domain=args.domain,
            metrics_path=str(metrics_path),
            importance_path=str(importance_path),
            output_dir=args.output_dir
        )
        return 0
    except Exception as e:
        logger.error(f"Figure generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
