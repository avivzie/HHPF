"""
Hypothesis testing and ablation studies for HHPF.

Tests the three main research hypotheses:
1. Feature Hypothesis: Hybrid features outperform uncertainty-only
2. Uncertainty Hypothesis: Semantic entropy/energy correlate with hallucinations  
3. Domain Hypothesis: Hallucination patterns differ across domains
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

from src.classifier.xgboost_model import HallucinationClassifier
from src.evaluation.metrics import MetricsCalculator
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypothesisTester:
    """Test research hypotheses through statistical analysis."""
    
    def __init__(self):
        """Initialize hypothesis tester."""
        self.calculator = MetricsCalculator()
    
    def ablation_study(
        self,
        features_df: pd.DataFrame,
        feature_groups: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Run ablation study comparing different feature combinations.
        
        Args:
            features_df: Full feature matrix
            feature_groups: Dict mapping model name to list of features to use
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Running ablation study...")
        
        results = []
        
        for model_name, features in feature_groups.items():
            logger.info(f"\nTraining {model_name}...")
            
            # Select features (filter out any that don't exist)
            available_features = [f for f in features if f in features_df.columns]
            
            if not available_features:
                logger.warning(f"  No features available for {model_name}, skipping")
                continue
            
            # Select features
            X = features_df[available_features]
            y = features_df['hallucination_label']
            
            # Split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train classifier
            classifier = HallucinationClassifier()
            classifier.feature_names = available_features
            classifier.train(X_train, y_train)
            
            # Evaluate
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)
            
            metrics = self.calculator.calculate_all_metrics(y_test, y_pred, y_proba)
            
            results.append({
                'model': model_name,
                'num_features': len(available_features),
                'auroc': metrics['auroc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'ece': metrics['ece']
            })
            
            logger.info(f"  Features used: {len(available_features)}")
            logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('auroc', ascending=False)
        
        return results_df
    
    def train_domain_specific_models(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Train separate models for each domain (RQ3).
        
        Args:
            features_df: Full feature matrix with all domains
            
        Returns:
            Dictionary mapping domain to model and metrics
        """
        logger.info("Training domain-specific models...")
        
        from src.features.feature_aggregator import FeatureAggregator
        aggregator = FeatureAggregator()
        feature_cols = aggregator.get_feature_columns(features_df)
        
        # Remove domain one-hot features for domain-specific models
        feature_cols = [f for f in feature_cols if not f.startswith('domain_')]
        
        domain_models = {}
        
        for domain in features_df['domain'].unique():
            logger.info(f"\nTraining model for {domain}...")
            
            # Filter domain data
            domain_data = features_df[features_df['domain'] == domain].copy()
            
            if len(domain_data) < 50:
                logger.warning(f"  Skipping {domain}: insufficient samples ({len(domain_data)})")
                continue
            
            X = domain_data[feature_cols]
            y = domain_data['hallucination_label']
            
            # Split
            from sklearn.model_selection import train_test_split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # If class distribution doesn't allow stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Train
            classifier = HallucinationClassifier()
            classifier.feature_names = feature_cols
            classifier.train(X_train, y_train)
            
            # Evaluate
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)
            
            metrics = self.calculator.calculate_all_metrics(y_test, y_pred, y_proba)
            
            # Get feature importance
            importance_df = classifier.get_feature_importance()
            
            domain_models[domain] = {
                'model': classifier,
                'metrics': metrics,
                'feature_importance': importance_df,
                'n_samples': len(domain_data)
            }
            
            logger.info(f"  Samples: {len(domain_data)}")
            logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        
        return domain_models
    
    def test_feature_importance_differences(
        self,
        domain_models: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Test if feature importance differs significantly across domains (RQ3).
        
        Args:
            domain_models: Domain-specific models with importance
            
        Returns:
            DataFrame with statistical test results
        """
        logger.info("Testing feature importance differences across domains...")
        
        # Collect importance scores for each feature across domains
        feature_importance_by_domain = {}
        
        for domain, model_data in domain_models.items():
            importance_df = model_data['feature_importance']
            
            for _, row in importance_df.iterrows():
                feature = row['feature']
                importance = row['importance']
                
                if feature not in feature_importance_by_domain:
                    feature_importance_by_domain[feature] = {}
                
                feature_importance_by_domain[feature][domain] = importance
        
        # Test each feature for significant differences across domains
        results = []
        
        for feature, domain_scores in feature_importance_by_domain.items():
            if len(domain_scores) < 2:
                continue
            
            # Get importance values for each domain
            domains = list(domain_scores.keys())
            scores = list(domain_scores.values())
            
            # Calculate variance
            mean_importance = np.mean(scores)
            std_importance = np.std(scores)
            cv = std_importance / mean_importance if mean_importance > 0 else 0
            
            # Range
            importance_range = max(scores) - min(scores)
            
            results.append({
                'feature': feature,
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'coefficient_variation': cv,
                'min_importance': min(scores),
                'max_importance': max(scores),
                'importance_range': importance_range,
                'num_domains': len(domains),
                'varies_across_domains': cv > 0.3  # High variation
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('coefficient_variation', ascending=False)
        
        # Log features with high cross-domain variation
        high_variation = results_df[results_df['varies_across_domains']]
        
        logger.info(f"\nFeatures with high cross-domain variation (CV > 0.3):")
        for _, row in high_variation.head(10).iterrows():
            logger.info(f"  {row['feature']}: CV={row['coefficient_variation']:.3f}, "
                       f"Range={row['importance_range']:.3f}")
        
        return results_df
    
    def feature_correlation_test(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Test correlation between features and hallucination label.
        
        Args:
            features_df: Feature matrix with hallucination_label
            feature_cols: List of feature columns to test
            
        Returns:
            DataFrame with correlation results and p-values
        """
        logger.info("Testing feature correlations...")
        
        results = []
        y = features_df['hallucination_label']
        
        for feature in feature_cols:
            if feature not in features_df.columns:
                continue
            
            x = features_df[feature].dropna()
            y_aligned = y[x.index]
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x, y_aligned)
            
            # Spearman correlation (non-parametric)
            spearman_r, spearman_p = stats.spearmanr(x, y_aligned)
            
            # Point-biserial correlation (for binary target)
            pointbiserial_r, pointbiserial_p = stats.pointbiserialr(y_aligned, x)
            
            results.append({
                'feature': feature,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pointbiserial_r': pointbiserial_r,
                'pointbiserial_p': pointbiserial_p,
                'significant_005': pearson_p < 0.05,
                'significant_001': pearson_p < 0.01
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('pearson_r', key=abs, ascending=False)
        
        # Log significant correlations
        sig_features = results_df[results_df['significant_005']]
        logger.info(f"\nFound {len(sig_features)} significantly correlated features (p < 0.05):")
        for _, row in sig_features.head(10).iterrows():
            logger.info(f"  {row['feature']}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.4f}")
        
        return results_df
    
    def domain_comparison_test(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Test if hallucination patterns differ significantly across domains.
        
        Args:
            features_df: Feature matrix with domain information
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing domain differences...")
        
        # Get hallucination rates by domain
        domain_stats = []
        
        for domain in features_df['domain'].unique():
            domain_data = features_df[features_df['domain'] == domain]
            hall_rate = domain_data['hallucination_label'].mean()
            n_samples = len(domain_data)
            
            domain_stats.append({
                'domain': domain,
                'hallucination_rate': hall_rate,
                'n_samples': n_samples
            })
        
        stats_df = pd.DataFrame(domain_stats)
        stats_df = stats_df.sort_values('hallucination_rate', ascending=False)
        
        logger.info("\nHallucination rates by domain:")
        for _, row in stats_df.iterrows():
            logger.info(f"  {row['domain']}: {row['hallucination_rate']:.1%} (n={row['n_samples']})")
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(
            features_df['domain'],
            features_df['hallucination_label']
        )
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        logger.info(f"\nChi-square test for domain independence:")
        logger.info(f"  χ² = {chi2:.4f}")
        logger.info(f"  p-value = {p_value:.6f}")
        logger.info(f"  df = {dof}")
        logger.info(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        # Pairwise domain comparisons (if significant)
        pairwise_results = []
        
        if p_value < 0.05:
            domains = features_df['domain'].unique()
            
            for i, domain1 in enumerate(domains):
                for domain2 in domains[i+1:]:
                    data1 = features_df[features_df['domain'] == domain1]['hallucination_label']
                    data2 = features_df[features_df['domain'] == domain2]['hallucination_label']
                    
                    # Two-proportion z-test
                    from statsmodels.stats.proportion import proportions_ztest
                    
                    counts = np.array([data1.sum(), data2.sum()])
                    nobs = np.array([len(data1), len(data2)])
                    
                    z_stat, p_val = proportions_ztest(counts, nobs)
                    
                    pairwise_results.append({
                        'domain1': domain1,
                        'domain2': domain2,
                        'z_statistic': z_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    })
        
        return {
            'domain_stats': stats_df,
            'chi_square': chi2,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < 0.05,
            'pairwise_comparisons': pd.DataFrame(pairwise_results) if pairwise_results else None
        }
    
    def knowledge_overshadowing_test(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Test the knowledge overshadowing hypothesis.
        
        Tests if rare entities (low popularity) increase hallucination risk.
        
        Args:
            features_df: Feature matrix with entity rarity features
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing knowledge overshadowing hypothesis...")
        
        if 'avg_entity_rarity' not in features_df.columns:
            logger.warning("avg_entity_rarity feature not found")
            return {}
        
        # Filter samples with entities
        df = features_df[features_df['num_entities'] > 0].copy()
        
        # Stratify by rarity (terciles)
        df['rarity_tercile'] = pd.qcut(
            df['avg_entity_rarity'],
            q=3,
            labels=['Low Rarity (Popular)', 'Medium Rarity', 'High Rarity (Rare)']
        )
        
        # Hallucination rate by rarity
        rarity_stats = df.groupby('rarity_tercile')['hallucination_label'].agg([
            'mean', 'count'
        ]).reset_index()
        rarity_stats.columns = ['rarity_group', 'hallucination_rate', 'n_samples']
        
        logger.info("\nHallucination rates by entity rarity:")
        for _, row in rarity_stats.iterrows():
            logger.info(f"  {row['rarity_group']}: {row['hallucination_rate']:.1%} (n={row['n_samples']})")
        
        # Trend test (Jonckheere-Terpstra test or correlation)
        correlation, p_value = stats.pearsonr(
            df['avg_entity_rarity'],
            df['hallucination_label']
        )
        
        logger.info(f"\nCorrelation between entity rarity and hallucinations:")
        logger.info(f"  Pearson r = {correlation:.4f}")
        logger.info(f"  p-value = {p_value:.6f}")
        logger.info(f"  Hypothesis supported: {'Yes' if (correlation > 0 and p_value < 0.05) else 'No'}")
        
        return {
            'rarity_stats': rarity_stats,
            'correlation': correlation,
            'p_value': p_value,
            'hypothesis_supported': correlation > 0 and p_value < 0.05
        }


def main():
    """Main entry point for hypothesis testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HHPF hypothesis tests")
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to feature matrix CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/results/hypothesis_tests.json',
        help='Output path for results'
    )
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Initialize tester
    tester = HypothesisTester()
    
    # 1. Ablation Study (RQ1 & RQ2)
    logger.info("\n" + "="*60)
    logger.info("RESEARCH QUESTION 1: Feature Hypothesis")
    logger.info("="*60)
    
    from src.features.feature_aggregator import FeatureAggregator
    aggregator = FeatureAggregator()
    all_features = aggregator.get_feature_columns(features_df)
    
    # Define feature groups for RQ1 and RQ2
    naive_features = [f for f in all_features if any(
        keyword in f for keyword in ['naive_max_prob', 'naive_perplexity', 'mean_logprob', 'min_logprob']
    )]
    
    semantic_features = [f for f in all_features if any(
        keyword in f for keyword in ['semantic_entropy', 'semantic_energy', 'num_semantic_clusters']
    )]
    
    contextual_features = [f for f in all_features if any(
        keyword in f for keyword in ['entity', 'rarity', 'token', 'lexical', 'parse', 'qtype', 'complexity']
    )]
    
    feature_groups = {
        'Baseline: Naive Confidence': naive_features,
        'Semantic Uncertainty': semantic_features,
        'Context Only': contextual_features,
        'Semantic + Context': semantic_features + contextual_features,
        'Full Model': all_features
    }
    
    ablation_results = tester.ablation_study(features_df, feature_groups)
    
    logger.info("\n✓ Ablation Study Results (RQ1 & RQ2):")
    print(ablation_results.to_string(index=False))
    
    # RQ2 Specific: Compare Semantic vs Naive
    logger.info("\n" + "="*60)
    logger.info("RESEARCH QUESTION 2: Semantic vs Naive Confidence")
    logger.info("="*60)
    
    naive_row = ablation_results[ablation_results['model'] == 'Baseline: Naive Confidence']
    semantic_row = ablation_results[ablation_results['model'] == 'Semantic Uncertainty']
    
    if not naive_row.empty and not semantic_row.empty:
        naive_auroc = naive_row['auroc'].values[0]
        semantic_auroc = semantic_row['auroc'].values[0]
        improvement = semantic_auroc - naive_auroc
        improvement_pct = (improvement / naive_auroc) * 100
        
        logger.info(f"\nSemantic Uncertainty vs Naive Confidence:")
        logger.info(f"  Naive AUROC:    {naive_auroc:.4f}")
        logger.info(f"  Semantic AUROC: {semantic_auroc:.4f}")
        logger.info(f"  Improvement:    {improvement:.4f} ({improvement_pct:+.1f}%)")
        logger.info(f"  Hypothesis supported: {'YES' if improvement > 0.05 else 'NO'} (threshold: +0.05 AUROC)")
    else:
        logger.warning("Could not compare semantic vs naive - missing features")
    
    # 2. Feature Correlation Analysis
    logger.info("\n" + "="*60)
    logger.info("HYPOTHESIS 2: Uncertainty Hypothesis")
    logger.info("="*60)
    
    correlation_results = tester.feature_correlation_test(features_df, all_features)
    
    # 3. Domain Comparison (RQ3)
    logger.info("\n" + "="*60)
    logger.info("RESEARCH QUESTION 3: Cross-Domain Variance")
    logger.info("="*60)
    
    # 3a. Overall domain comparison
    domain_results = tester.domain_comparison_test(features_df)
    
    # 3b. Train domain-specific models
    logger.info("\nTraining domain-specific models...")
    domain_models = tester.train_domain_specific_models(features_df)
    
    # 3c. Compare feature importance across domains
    logger.info("\nAnalyzing feature importance differences...")
    importance_diff = tester.test_feature_importance_differences(domain_models)
    
    # 3d. Compare model performance across domains
    logger.info("\n✓ Domain-Specific Model Performance:")
    for domain, model_data in domain_models.items():
        logger.info(f"  {domain}: AUROC={model_data['metrics']['auroc']:.4f}, "
                   f"n={model_data['n_samples']}")
    
    # Save domain-specific models
    from pathlib import Path
    from src.utils import ensure_dir
    
    models_dir = Path(args.output).parent / "domain_models"
    ensure_dir(models_dir)
    
    for domain, model_data in domain_models.items():
        model_path = models_dir / f"xgboost_{domain}.pkl"
        model_data['model'].save_model(str(model_path))
        
        # Save feature importance
        importance_path = models_dir / f"feature_importance_{domain}.csv"
        model_data['feature_importance'].to_csv(importance_path, index=False)
    
    logger.info(f"\n✓ Domain-specific models saved to {models_dir}")
    
    # 4. Knowledge Overshadowing
    logger.info("\n" + "="*60)
    logger.info("ADDITIONAL TEST: Knowledge Overshadowing")
    logger.info("="*60)
    
    knowledge_results = tester.knowledge_overshadowing_test(features_df)
    
    # Save results
    import json
    from pathlib import Path
    from src.utils import ensure_dir
    
    ensure_dir(Path(args.output).parent)
    
    # Prepare domain-specific metrics
    domain_metrics_summary = {}
    domain_importance_summary = {}
    
    for domain, model_data in domain_models.items():
        domain_metrics_summary[domain] = {
            'auroc': float(model_data['metrics']['auroc']),
            'accuracy': float(model_data['metrics']['accuracy']),
            'n_samples': int(model_data['n_samples'])
        }
        
        # Top 10 features per domain
        top_features = model_data['feature_importance'].head(10)
        domain_importance_summary[domain] = top_features.to_dict(orient='records')
    
    # Save comprehensive summary
    summary = {
        'research_question_1': {
            'description': 'Feature Hypothesis: Hybrid features outperform baselines',
            'ablation_study': ablation_results.to_dict(orient='records')
        },
        'research_question_2': {
            'description': 'Semantic Uncertainty vs Naive Confidence',
            'comparison': {
                'naive_auroc': float(ablation_results[ablation_results['model'] == 'Baseline: Naive Confidence']['auroc'].values[0]) if 'Baseline: Naive Confidence' in ablation_results['model'].values else None,
                'semantic_auroc': float(ablation_results[ablation_results['model'] == 'Semantic Uncertainty']['auroc'].values[0]) if 'Semantic Uncertainty' in ablation_results['model'].values else None,
            }
        },
        'research_question_3': {
            'description': 'Cross-Domain Variance in Hallucination Signatures',
            'domain_comparison': {
                'domain_stats': domain_results['domain_stats'].to_dict(orient='records'),
                'chi_square': float(domain_results['chi_square']),
                'p_value': float(domain_results['p_value']),
                'significant': bool(domain_results['significant'])
            },
            'domain_specific_performance': domain_metrics_summary,
            'feature_importance_differences': importance_diff.head(20).to_dict(orient='records'),
            'top_features_by_domain': domain_importance_summary
        },
        'additional_analysis': {
            'top_correlated_features': correlation_results.head(20).to_dict(orient='records'),
            'knowledge_overshadowing': {
                'correlation': float(knowledge_results.get('correlation', 0)),
                'p_value': float(knowledge_results.get('p_value', 1)),
                'hypothesis_supported': bool(knowledge_results.get('hypothesis_supported', False))
            }
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    correlation_results.to_csv(
        Path(args.output).parent / 'feature_correlations.csv',
        index=False
    )
    
    ablation_results.to_csv(
        Path(args.output).parent / 'ablation_study_results.csv',
        index=False
    )
    
    importance_diff.to_csv(
        Path(args.output).parent / 'domain_feature_importance_differences.csv',
        index=False
    )
    
    # Save domain-specific metrics
    domain_metrics_df = pd.DataFrame.from_dict(domain_metrics_summary, orient='index')
    domain_metrics_df.to_csv(
        Path(args.output).parent / 'domain_specific_metrics.csv'
    )
    
    logger.info(f"\n✓ All hypothesis test results saved to {Path(args.output).parent}")
    logger.info(f"  - Main summary: {args.output}")
    logger.info(f"  - Ablation study: ablation_study_results.csv")
    logger.info(f"  - Feature correlations: feature_correlations.csv")
    logger.info(f"  - Domain importance differences: domain_feature_importance_differences.csv")
    logger.info(f"  - Domain metrics: domain_specific_metrics.csv")


if __name__ == "__main__":
    main()
