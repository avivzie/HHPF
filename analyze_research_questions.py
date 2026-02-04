"""
Comprehensive analysis script for all three research questions.

This script runs complete analysis to answer:
RQ1: Feature Hypothesis (hybrid features outperform baselines)
RQ2: Semantic Uncertainty vs Naive Confidence
RQ3: Cross-Domain Variance in Hallucination Signatures
"""

import argparse
import pandas as pd
import logging
from pathlib import Path

from src.evaluation.hypothesis_testing import HypothesisTester
from src.evaluation.visualization import HHPFVisualizer
from src.utils import ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_all_research_questions(
    features_path: str,
    output_dir: str = "outputs/research_questions"
):
    """
    Run comprehensive analysis for all research questions.
    
    Args:
        features_path: Path to combined feature matrix (all domains)
        output_dir: Output directory for results
    """
    logger.info("="*80)
    logger.info("COMPREHENSIVE RESEARCH QUESTIONS ANALYSIS")
    logger.info("="*80)
    
    # Load features
    logger.info(f"\nLoading features from {features_path}")
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} samples across {features_df['domain'].nunique()} domains")
    
    # Initialize tester and visualizer
    tester = HypothesisTester()
    viz = HHPFVisualizer(output_dir=f"{output_dir}/figures")
    
    # Prepare feature groups
    from src.features.feature_aggregator import FeatureAggregator
    aggregator = FeatureAggregator()
    all_features = aggregator.get_feature_columns(features_df)
    
    # Define feature groups
    naive_features = [f for f in all_features if any(
        keyword in f for keyword in ['naive_max_prob', 'naive_perplexity', 'mean_logprob', 'min_logprob']
    )]
    
    semantic_features = [f for f in all_features if any(
        keyword in f for keyword in ['semantic_entropy', 'semantic_energy', 'num_semantic_clusters']
    )]
    
    contextual_features = [f for f in all_features if any(
        keyword in f for keyword in ['entity', 'rarity', 'token', 'lexical', 'parse', 'qtype', 'complexity']
    )]
    
    # ========================================================================
    # RESEARCH QUESTION 1: Feature Hypothesis
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RESEARCH QUESTION 1: Feature Hypothesis")
    logger.info("Do hybrid features (semantic + contextual) outperform baselines?")
    logger.info("="*80)
    
    feature_groups_rq1 = {
        'Baseline: Naive Confidence': naive_features,
        'Semantic Uncertainty': semantic_features,
        'Context Only': contextual_features,
        'Semantic + Context': semantic_features + contextual_features,
        'Full Model (All Features)': all_features
    }
    
    ablation_results = tester.ablation_study(features_df, feature_groups_rq1)
    
    logger.info("\n✓ Ablation Study Results:")
    print(ablation_results[['model', 'num_features', 'auroc', 'accuracy', 'f1']].to_string(index=False))
    
    # Visualize
    models_data_rq1 = {}
    for _, row in ablation_results.iterrows():
        models_data_rq1[row['model']] = {
            'auroc': row['auroc'],
            'roc_curve': {'fpr': [], 'tpr': [], 'thresholds': []},  # Simplified
            'arc': {'rejection_rates': [], 'accuracies': [], 'auc_arc': 0}
        }
    
    # ========================================================================
    # RESEARCH QUESTION 2: Semantic vs Naive Confidence
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RESEARCH QUESTION 2: Semantic Uncertainty vs Naive Confidence")
    logger.info("Does Semantic Entropy outperform naive confidence metrics?")
    logger.info("="*80)
    
    naive_row = ablation_results[ablation_results['model'] == 'Baseline: Naive Confidence']
    semantic_row = ablation_results[ablation_results['model'] == 'Semantic Uncertainty']
    
    if not naive_row.empty and not semantic_row.empty:
        naive_auroc = naive_row['auroc'].values[0]
        semantic_auroc = semantic_row['auroc'].values[0]
        improvement = semantic_auroc - naive_auroc
        improvement_pct = (improvement / naive_auroc) * 100
        
        logger.info(f"\n✓ Comparison Results:")
        logger.info(f"  Naive Confidence AUROC:    {naive_auroc:.4f}")
        logger.info(f"  Semantic Uncertainty AUROC: {semantic_auroc:.4f}")
        logger.info(f"  Absolute Improvement:       {improvement:+.4f}")
        logger.info(f"  Relative Improvement:       {improvement_pct:+.1f}%")
        logger.info(f"\n  Hypothesis: Semantic > Naive")
        logger.info(f"  Result: {'✓ SUPPORTED' if improvement > 0.05 else '✗ NOT SUPPORTED'} (threshold: +0.05 AUROC)")
        
        # Visualize comparison
        viz.plot_semantic_vs_naive_comparison(
            ablation_results,
            filename='rq2_semantic_vs_naive'
        )
    else:
        logger.warning("Could not perform RQ2 comparison - missing baseline features")
    
    # ========================================================================
    # RESEARCH QUESTION 3: Cross-Domain Variance
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RESEARCH QUESTION 3: Cross-Domain Variance")
    logger.info("Do hallucination signatures differ significantly across domains?")
    logger.info("="*80)
    
    # 3a. Domain comparison test
    domain_results = tester.domain_comparison_test(features_df)
    
    logger.info(f"\n✓ Overall Domain Comparison:")
    logger.info(f"  Chi-square statistic: {domain_results['chi_square']:.4f}")
    logger.info(f"  p-value: {domain_results['p_value']:.6f}")
    logger.info(f"  Hypothesis: Domains differ significantly")
    logger.info(f"  Result: {'✓ SUPPORTED' if domain_results['significant'] else '✗ NOT SUPPORTED'} (α=0.05)")
    
    # 3b. Train domain-specific models
    logger.info("\n✓ Training domain-specific models...")
    domain_models = tester.train_domain_specific_models(features_df)
    
    logger.info("\n✓ Domain-Specific Performance:")
    for domain, model_data in domain_models.items():
        logger.info(f"  {domain:15s}: AUROC={model_data['metrics']['auroc']:.4f}, "
                   f"Accuracy={model_data['metrics']['accuracy']:.4f}, "
                   f"n={model_data['n_samples']}")
    
    # 3c. Test feature importance differences
    logger.info("\n✓ Analyzing feature importance differences across domains...")
    importance_diff = tester.test_feature_importance_differences(domain_models)
    
    logger.info("\n✓ Features with High Cross-Domain Variation:")
    high_var = importance_diff[importance_diff['varies_across_domains']].head(10)
    for _, row in high_var.iterrows():
        logger.info(f"  {row['feature']:30s}: CV={row['coefficient_variation']:.3f}, "
                   f"Range={row['importance_range']:.3f}")
    
    # Visualizations for RQ3
    viz.plot_domain_specific_auroc(
        domain_models,
        filename='rq3_domain_auroc'
    )
    
    viz.plot_domain_feature_importance_heatmap(
        domain_models,
        top_k=15,
        filename='rq3_domain_feature_heatmap'
    )
    
    # ========================================================================
    # SAVE ALL RESULTS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    ensure_dir(output_dir)
    
    # Save ablation results
    ablation_results.to_csv(f"{output_dir}/rq1_ablation_study.csv", index=False)
    logger.info(f"✓ RQ1 results: {output_dir}/rq1_ablation_study.csv")
    
    # Save RQ2 comparison
    if not naive_row.empty and not semantic_row.empty:
        rq2_results = pd.DataFrame([
            {
                'metric': 'AUROC',
                'naive_confidence': naive_auroc,
                'semantic_uncertainty': semantic_auroc,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'hypothesis_supported': improvement > 0.05
            }
        ])
        rq2_results.to_csv(f"{output_dir}/rq2_semantic_vs_naive.csv", index=False)
        logger.info(f"✓ RQ2 results: {output_dir}/rq2_semantic_vs_naive.csv")
    
    # Save RQ3 domain results
    domain_metrics_df = pd.DataFrame([
        {
            'domain': domain,
            'auroc': model_data['metrics']['auroc'],
            'accuracy': model_data['metrics']['accuracy'],
            'precision': model_data['metrics']['precision'],
            'recall': model_data['metrics']['recall'],
            'f1': model_data['metrics']['f1'],
            'n_samples': model_data['n_samples']
        }
        for domain, model_data in domain_models.items()
    ])
    domain_metrics_df.to_csv(f"{output_dir}/rq3_domain_metrics.csv", index=False)
    logger.info(f"✓ RQ3 metrics: {output_dir}/rq3_domain_metrics.csv")
    
    # Save feature importance differences
    importance_diff.to_csv(f"{output_dir}/rq3_feature_importance_differences.csv", index=False)
    logger.info(f"✓ RQ3 importance: {output_dir}/rq3_feature_importance_differences.csv")
    
    # Save domain-specific models
    models_dir = Path(output_dir) / "domain_models"
    ensure_dir(models_dir)
    
    for domain, model_data in domain_models.items():
        model_path = models_dir / f"xgboost_{domain}.pkl"
        model_data['model'].save_model(str(model_path))
        
        importance_path = models_dir / f"feature_importance_{domain}.csv"
        model_data['feature_importance'].to_csv(importance_path, index=False)
    
    logger.info(f"✓ Domain models: {models_dir}/")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE - SUMMARY")
    logger.info("="*80)
    
    logger.info("\n📊 RESEARCH QUESTION 1: Feature Hypothesis")
    logger.info(f"   Best Model: {ablation_results.iloc[0]['model']}")
    logger.info(f"   Best AUROC: {ablation_results.iloc[0]['auroc']:.4f}")
    
    if not naive_row.empty and not semantic_row.empty:
        logger.info("\n📊 RESEARCH QUESTION 2: Semantic vs Naive")
        logger.info(f"   Semantic outperforms Naive: {'YES' if improvement > 0.05 else 'NO'}")
        logger.info(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    logger.info("\n📊 RESEARCH QUESTION 3: Cross-Domain Variance")
    logger.info(f"   Domains differ significantly: {'YES' if domain_results['significant'] else 'NO'}")
    logger.info(f"   Number of domain-specific models: {len(domain_models)}")
    logger.info(f"   Features varying across domains: {importance_diff['varies_across_domains'].sum()}")
    
    logger.info("\n📁 All outputs saved to:")
    logger.info(f"   Results: {output_dir}/")
    logger.info(f"   Figures: {output_dir}/figures/")
    logger.info(f"   Models: {output_dir}/domain_models/")
    
    logger.info("\n✅ Ready for thesis writing!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze all three research questions for HHPF thesis"
    )
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to combined feature matrix (all domains)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/research_questions',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_all_research_questions(
            features_path=args.features,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
