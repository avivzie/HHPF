"""
Generate publication-quality figures for thesis Results section.

Figures:
1. RQ1: Ablation study comparison (bar chart with error bars)
2. RQ2: Semantic vs Naive comparison (grouped bar chart)
3. RQ3a: Hallucination rate distribution (stacked bar chart)
4. RQ3b: Domain-specific AUROC (bar chart)
5. RQ3c: Feature importance heatmap
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_rq1_ablation(aggregated_df, output_dir):
    """
    Figure 1: RQ1 Ablation Study Results
    Bar chart showing AUROC for each feature subset with error bars
    """
    logger.info("\nGenerating Figure 1: RQ1 Ablation Study...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by AUROC
    df_sorted = aggregated_df.sort_values('auroc_mean', ascending=True)
    
    # Create bar chart
    x = np.arange(len(df_sorted))
    bars = ax.barh(x, df_sorted['auroc_mean'], 
                   xerr=df_sorted['auroc_std'],
                   capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Color coding
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(x)
    ax.set_yticklabels(df_sorted['feature_subset'])
    ax.set_xlabel('Test AUROC (mean ± std across 5 domains)')
    ax.set_title('RQ1: Feature Ablation Study Results', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0.4, 0.85)
    
    # Add value labels
    for i, (auroc, std) in enumerate(zip(df_sorted['auroc_mean'], df_sorted['auroc_std'])):
        ax.text(auroc + std + 0.01, i, f'{auroc:.3f}±{std:.3f}', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save both PDF and PNG
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'rq1_ablation_comparison.{ext}'
        plt.savefig(path, bbox_inches='tight')
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_rq2_semantic_vs_naive(per_domain_df, output_dir):
    """
    Figure 2: RQ2 Semantic vs Naive Comparison
    Grouped bar chart showing per-domain comparison
    """
    logger.info("\nGenerating Figure 2: RQ2 Semantic vs Naive...")
    
    # Filter for Semantic-Only and Naive-Only
    semantic_df = per_domain_df[per_domain_df['feature_subset'] == 'Semantic-Only'][['domain', 'auroc']]
    naive_df = per_domain_df[per_domain_df['feature_subset'] == 'Naive-Only'][['domain', 'auroc']]
    
    domains = semantic_df['domain'].values
    semantic_aurocs = semantic_df['auroc'].values
    naive_aurocs = naive_df['auroc'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, naive_aurocs, width, label='Naive Confidence',
                   color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, semantic_aurocs, width, label='Semantic Uncertainty',
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Test AUROC')
    ax.set_title('RQ2: Semantic Uncertainty vs Naive Confidence by Domain', 
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 0.85)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'rq2_semantic_vs_naive.{ext}'
        plt.savefig(path, bbox_inches='tight')
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_rq3a_hallucination_rates(rq3a_path, output_dir):
    """
    Figure 3: RQ3a Hallucination Rate Distribution
    Stacked bar chart showing hallucination vs faithful samples
    """
    logger.info("\nGenerating Figure 3: RQ3a Hallucination Rates...")
    
    df = pd.read_csv(rq3a_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.6
    
    bars1 = ax.bar(x, df['hallucinations'], width, label='Hallucinations',
                   color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, df['faithful'], width, bottom=df['hallucinations'],
                   label='Faithful', color='#2ca02c', alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Number of Samples')
    ax.set_title('RQ3a: Hallucination Rate Distribution Across Domains', 
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df['domain'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for i, row in df.iterrows():
        hall_pct = row['hallucination_rate'] * 100
        ax.text(i, row['total']/2, f'{hall_pct:.1f}%', 
               ha='center', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'rq3a_hallucination_rates.{ext}'
        plt.savefig(path, bbox_inches='tight')
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_rq3b_domain_auroc(aggregated_df, per_domain_df, output_dir):
    """
    Figure 4: RQ3b Domain-Specific AUROC
    Bar chart showing Full model AUROC for each domain
    """
    logger.info("\nGenerating Figure 4: RQ3b Domain AUROC...")
    
    # Get Full model AUROC for each domain
    full_df = per_domain_df[per_domain_df['feature_subset'] == 'Full'][['domain', 'auroc']]
    full_df = full_df.sort_values('auroc', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(full_df))
    bars = ax.barh(x, full_df['auroc'], 
                   alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Color bars by performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(full_df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(x)
    ax.set_yticklabels(full_df['domain'])
    ax.set_xlabel('Test AUROC (Full Model)')
    ax.set_title('RQ3b: Domain-Specific Performance', fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 0.85)
    
    # Add mean line
    mean_auroc = aggregated_df[aggregated_df['feature_subset'] == 'Full']['auroc_mean'].values[0]
    ax.axvline(mean_auroc, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_auroc:.3f}', alpha=0.7)
    ax.legend()
    
    # Add value labels
    for i, auroc in enumerate(full_df['auroc']):
        ax.text(auroc + 0.01, i, f'{auroc:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'rq3b_domain_auroc.{ext}'
        plt.savefig(path, bbox_inches='tight')
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def plot_rq3c_feature_heatmap(ablation_dir, output_dir, top_k=15):
    """
    Figure 5: RQ3c Feature Importance Heatmap
    Heatmap showing top features × domains
    """
    logger.info(f"\nGenerating Figure 5: RQ3c Feature Importance Heatmap (top {top_k})...")
    
    domains = ['math', 'is_agents', 'psychology', 'medicine', 'finance']
    importance_data = []
    
    # Load feature importance from each domain
    for domain in domains:
        importance_path = Path(ablation_dir) / f"{domain}_feature_importance.csv"
        df = pd.read_csv(importance_path)
        df['domain'] = domain
        importance_data.append(df)
    
    # Combine and pivot
    combined = pd.concat(importance_data)
    
    # Get top features by mean importance across domains
    top_features = combined.groupby('feature')['importance'].mean().nlargest(top_k).index
    
    # Create pivot table
    pivot = combined[combined['feature'].isin(top_features)].pivot(
        index='feature', columns='domain', values='importance'
    )
    
    # Reorder columns
    pivot = pivot[domains]
    
    # Sort by mean importance
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False).drop('mean', axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
               linewidths=0.5, cbar_kws={'label': 'Importance Score'},
               ax=ax)
    
    ax.set_title(f'RQ3c: Top {top_k} Feature Importance Across Domains', 
                fontweight='bold', pad=15)
    ax.set_xlabel('Domain')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = Path(output_dir) / f'rq3c_feature_importance_heatmap.{ext}'
        plt.savefig(path, bbox_inches='tight')
        logger.info(f"  ✓ Saved {path}")
    
    plt.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready figures"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='outputs/research_questions',
        help='Directory containing analysis results'
    )
    parser.add_argument(
        '--ablation-dir',
        type=str,
        default='outputs/ablation',
        help='Directory containing ablation results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/research_questions/figures',
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("="*80)
        logger.info("GENERATING THESIS-READY FIGURES")
        logger.info("="*80)
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data
        aggregated_path = Path(args.results_dir) / 'rq1_rq2_aggregated_results.csv'
        per_domain_path = Path(args.results_dir) / 'per_domain_ablation_breakdown.csv'
        rq3a_path = Path(args.results_dir) / 'rq3a_hallucination_rates.csv'
        
        if not aggregated_path.exists():
            logger.error(f"Aggregated results not found: {aggregated_path}")
            return
        
        aggregated_df = pd.read_csv(aggregated_path)
        per_domain_df = pd.read_csv(per_domain_path)
        
        # Generate figures
        plot_rq1_ablation(aggregated_df, args.output_dir)
        plot_rq2_semantic_vs_naive(per_domain_df, args.output_dir)
        plot_rq3a_hallucination_rates(rq3a_path, args.output_dir)
        plot_rq3b_domain_auroc(aggregated_df, per_domain_df, args.output_dir)
        plot_rq3c_feature_heatmap(args.ablation_dir, args.output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("ALL FIGURES GENERATED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"\nFigures saved to: {args.output_dir}")
        logger.info("  • rq1_ablation_comparison.pdf/png")
        logger.info("  • rq2_semantic_vs_naive.pdf/png")
        logger.info("  • rq3a_hallucination_rates.pdf/png")
        logger.info("  • rq3b_domain_auroc.pdf/png")
        logger.info("  • rq3c_feature_importance_heatmap.pdf/png")
        
    except Exception as e:
        logger.error(f"Figure generation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
