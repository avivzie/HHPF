"""
Aggregate per-domain ablation results for cross-domain comparison.

This script combines ablation results from all 5 domains and computes
summary statistics (mean ± std) for each feature subset.

Output: outputs/research_questions/rq1_rq2_aggregated_results.csv
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def aggregate_ablation_results(ablation_dir: str, output_dir: str):
    """
    Aggregate ablation results across all domains.
    
    Args:
        ablation_dir: Directory containing per-domain ablation CSVs
        output_dir: Output directory for aggregated results
    """
    logger.info("="*80)
    logger.info("AGGREGATING ABLATION RESULTS ACROSS DOMAINS")
    logger.info("="*80)
    
    # Load all domain ablation results
    domains = ['math', 'is_agents', 'psychology', 'medicine', 'finance']
    all_results = []
    
    for domain in domains:
        ablation_path = Path(ablation_dir) / f"{domain}_ablation_results.csv"
        
        if not ablation_path.exists():
            logger.warning(f"Ablation results not found for {domain}: {ablation_path}")
            continue
        
        df = pd.read_csv(ablation_path)
        df['domain'] = domain
        all_results.append(df)
        logger.info(f"✓ Loaded {len(df)} feature subsets for {domain}")
    
    if not all_results:
        logger.error("No ablation results found!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"\nTotal records: {len(combined_df)} across {len(all_results)} domains")

    has_cv = 'auroc_cv_mean' in combined_df.columns

    # Base aggregation (single-split)
    agg_dict = {
        'n_features': 'first',
        'auroc': ['mean', 'std', 'min', 'max', 'count'],
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std']
    }
    if has_cv:
        agg_dict.update({
            'auroc_cv_mean': ['mean', 'std'],
            'accuracy_cv_mean': ['mean', 'std'],
            'precision_cv_mean': ['mean', 'std'],
            'recall_cv_mean': ['mean', 'std'],
            'f1_cv_mean': ['mean', 'std'],
        })

    aggregated = combined_df.groupby('feature_subset').agg(agg_dict).reset_index()

    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                          for col in aggregated.columns.values]

    # Rename for clarity
    aggregated = aggregated.rename(columns={
        'n_features_first': 'n_features',
        'auroc_count': 'n_domains',
        'auroc_mean': 'auroc_mean',
        'auroc_std': 'auroc_std',
        'auroc_min': 'auroc_min',
        'auroc_max': 'auroc_max'
    })

    column_order = [
        'feature_subset', 'n_domains', 'n_features',
        'auroc_mean', 'auroc_std', 'auroc_min', 'auroc_max',
        'accuracy_mean', 'accuracy_std',
        'precision_mean', 'precision_std',
        'recall_mean', 'recall_std',
        'f1_mean', 'f1_std'
    ]
    if has_cv:
        aggregated = aggregated.rename(columns={
            'auroc_cv_mean_mean': 'auroc_cv_avg',
            'auroc_cv_mean_std': 'auroc_cv_std_avg',
            'accuracy_cv_mean_mean': 'accuracy_cv_avg',
            'accuracy_cv_mean_std': 'accuracy_cv_std_avg',
            'precision_cv_mean_mean': 'precision_cv_avg',
            'precision_cv_mean_std': 'precision_cv_std_avg',
            'recall_cv_mean_mean': 'recall_cv_avg',
            'recall_cv_mean_std': 'recall_cv_std_avg',
            'f1_cv_mean_mean': 'f1_cv_avg',
            'f1_cv_mean_std': 'f1_cv_std_avg',
        })
        column_order.extend([
            'auroc_cv_avg', 'auroc_cv_std_avg',
            'accuracy_cv_avg', 'accuracy_cv_std_avg',
            'precision_cv_avg', 'precision_cv_std_avg',
            'recall_cv_avg', 'recall_cv_std_avg',
            'f1_cv_avg', 'f1_cv_std_avg'
        ])

    aggregated = aggregated[[c for c in column_order if c in aggregated.columns]]
    aggregated = aggregated.sort_values('auroc_mean', ascending=False)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'rq1_rq2_aggregated_results.csv'
    aggregated.to_csv(output_path, index=False)

    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nResults saved to {output_path}")
    logger.info("\nAggregated Results:")
    logger.info("\n" + aggregated[['feature_subset', 'n_domains', 'auroc_mean', 'auroc_std']].to_string(index=False))

    # CV comparison: single-split vs CV per domain
    if has_cv:
        comparison_rows = []
        for _, row in combined_df.iterrows():
            comparison_rows.append({
                'domain': row['domain'],
                'feature_subset': row['feature_subset'],
                'auroc_single': row['auroc'],
                'auroc_cv_mean': row['auroc_cv_mean'],
                'auroc_cv_std': row['auroc_cv_std'],
                'auroc_diff': row['auroc'] - row['auroc_cv_mean'],
            })
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = Path(output_dir) / 'rq1_rq2_cv_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nCV comparison saved to {comparison_path}")

    per_domain_path = Path(output_dir) / 'per_domain_ablation_breakdown.csv'
    combined_df.to_csv(per_domain_path, index=False)
    logger.info(f"\nPer-domain breakdown saved to {per_domain_path}")

    return aggregated, combined_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results across domains"
    )
    parser.add_argument(
        '--ablation-dir',
        type=str,
        default='outputs/ablation',
        help='Directory containing per-domain ablation CSVs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/research_questions',
        help='Output directory for aggregated results'
    )
    
    args = parser.parse_args()
    
    try:
        aggregate_ablation_results(
            ablation_dir=args.ablation_dir,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Aggregation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
