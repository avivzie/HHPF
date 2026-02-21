"""
Statistical tests for research questions RQ1, RQ2, and RQ3.

RQ1: Paired t-test (Semantic+Context vs Naive-Only)
RQ2: Paired t-test (Semantic-Only vs Naive-Only, one-tailed)
RQ3a: Chi-square test (hallucination rate differences)
RQ3c: Feature importance variability (coefficient of variation)
"""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Number of primary tests for multiple comparison correction (RQ1, RQ2, RQ3a)
N_PRIMARY_TESTS = 3


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval using t-distribution.
    Returns: (mean, lower_bound, upper_bound)
    """
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return float(np.mean(data)), float(np.mean(data)), float(np.mean(data))
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)
    return float(mean), float(ci[0]), float(ci[1])


def bonferroni_correction(p_values, alpha=0.05):
    """
    Bonferroni correction: reject H0 if p < alpha/n.
    Returns: (corrected_alpha, list of significance per test)
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    reject = [p < corrected_alpha for p in p_values]
    return corrected_alpha, reject


def fdr_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns: (p_corrected array, reject array)
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        logger.warning("statsmodels not installed; FDR correction skipped. pip install statsmodels")
        return list(p_values), [False] * len(p_values)
    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return p_corrected.tolist(), reject.tolist()


def paired_t_test_rq1(per_domain_df):
    """
    RQ1: Do hybrid features (Semantic+Context) outperform Naive-Only baseline?
    
    H0: Semantic+Context AUROC = Naive-Only AUROC
    H1: Semantic+Context AUROC ≠ Naive-Only AUROC (two-tailed)
    """
    logger.info("\n" + "="*80)
    logger.info("RQ1: HYBRID FEATURES VS NAIVE BASELINE")
    logger.info("="*80)
    
    # Extract AUROCs for each domain
    hybrid = per_domain_df[per_domain_df['feature_subset'] == 'Semantic+Context']['auroc'].values
    naive = per_domain_df[per_domain_df['feature_subset'] == 'Naive-Only']['auroc'].values
    domains = per_domain_df[per_domain_df['feature_subset'] == 'Semantic+Context']['domain'].values
    
    # Ensure paired samples
    assert len(hybrid) == len(naive) == 5, "Should have 5 domain pairs"
    
    logger.info("\nPer-domain AUROCs:")
    for i, domain in enumerate(domains):
        logger.info(f"  {domain:15s}: Semantic+Context={hybrid[i]:.4f}, Naive-Only={naive[i]:.4f}, Δ={hybrid[i]-naive[i]:+.4f}")
    
    # Paired t-test (two-tailed)
    t_stat, p_value = stats.ttest_rel(hybrid, naive)
    diff = hybrid - naive

    # Effect size (Cohen's d for paired samples)
    cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # 95% CI for mean difference
    mean_diff, ci_low, ci_high = calculate_confidence_interval(diff, confidence=0.95)
    ci_95 = [ci_low, ci_high]

    # Per-domain CIs for difference (each domain is 1 obs so CI is degenerate; store point estimate)
    per_domain_cis = {}
    for i, d in enumerate(domains):
        per_domain_cis[d] = {'diff': float(diff[i]), 'ci': [float(diff[i]), float(diff[i])]}

    logger.info(f"\nStatistical Test Results:")
    logger.info(f"  Mean Δ AUROC:        {np.mean(diff):+.4f}")
    logger.info(f"  95% CI:              [{ci_low:.4f}, {ci_high:.4f}]")
    logger.info(f"  Std Δ AUROC:         {np.std(diff, ddof=1):.4f}")
    logger.info(f"  t-statistic:         {t_stat:.4f}")
    logger.info(f"  p-value (two-tailed): {p_value:.6f}")
    logger.info(f"  Cohen's d:           {cohen_d:.4f}")
    logger.info(f"  Degrees of freedom:  {len(hybrid)-1}")

    if p_value < 0.05:
        logger.info(f"\n✅ RESULT: Hybrid features SIGNIFICANTLY outperform Naive baseline (p < 0.05)")
    else:
        logger.info(f"\n❌ RESULT: No significant difference between Hybrid and Naive (p ≥ 0.05)")

    return {
        'test': 'RQ1: Paired t-test (Semantic+Context vs Naive-Only)',
        'hybrid_mean': float(np.mean(hybrid)),
        'naive_mean': float(np.mean(naive)),
        'mean_improvement': mean_diff,
        'ci_95': ci_95,
        'std_improvement': float(np.std(diff, ddof=1)),
        't_statistic': float(t_stat),
        'p_value_raw': float(p_value),
        'cohen_d': float(cohen_d),
        'significant_raw': p_value < 0.05,
        'domains': domains.tolist(),
        'hybrid_aurocs': hybrid.tolist(),
        'naive_aurocs': naive.tolist(),
        'per_domain_cis': per_domain_cis,
    }


def paired_t_test_rq2(per_domain_df):
    """
    RQ2: Does Semantic-Only outperform Naive-Only?
    
    H0: Semantic-Only AUROC = Naive-Only AUROC
    H1: Semantic-Only AUROC > Naive-Only AUROC (one-tailed)
    """
    logger.info("\n" + "="*80)
    logger.info("RQ2: SEMANTIC UNCERTAINTY VS NAIVE CONFIDENCE")
    logger.info("="*80)
    
    # Extract AUROCs for each domain
    semantic = per_domain_df[per_domain_df['feature_subset'] == 'Semantic-Only']['auroc'].values
    naive = per_domain_df[per_domain_df['feature_subset'] == 'Naive-Only']['auroc'].values
    domains = per_domain_df[per_domain_df['feature_subset'] == 'Semantic-Only']['domain'].values
    
    # Ensure paired samples
    assert len(semantic) == len(naive) == 5, "Should have 5 domain pairs"
    
    logger.info("\nPer-domain AUROCs:")
    for i, domain in enumerate(domains):
        logger.info(f"  {domain:15s}: Semantic-Only={semantic[i]:.4f}, Naive-Only={naive[i]:.4f}, Δ={semantic[i]-naive[i]:+.4f}")
    
    # Paired t-test (one-tailed: greater)
    t_stat, p_value_two_tailed = stats.ttest_rel(semantic, naive)
    p_value = p_value_two_tailed / 2 if t_stat > 0 else 1 - (p_value_two_tailed / 2)
    diff = semantic - naive

    # Effect size (Cohen's d for paired samples)
    cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # 95% CI for mean difference (two-sided CI; one-tailed test interpretation in text)
    mean_diff, ci_low, ci_high = calculate_confidence_interval(diff, confidence=0.95)
    ci_95 = [ci_low, ci_high]

    logger.info(f"\nStatistical Test Results:")
    logger.info(f"  Mean Δ AUROC:        {np.mean(diff):+.4f}")
    logger.info(f"  95% CI:              [{ci_low:.4f}, {ci_high:.4f}]")
    logger.info(f"  Std Δ AUROC:         {np.std(diff, ddof=1):.4f}")
    logger.info(f"  t-statistic:         {t_stat:.4f}")
    logger.info(f"  p-value (one-tailed): {p_value:.6f}")
    logger.info(f"  Cohen's d:           {cohen_d:.4f}")
    logger.info(f"  Degrees of freedom:  {len(semantic)-1}")

    if p_value < 0.05 and t_stat > 0:
        logger.info(f"\n✅ RESULT: Semantic uncertainty SIGNIFICANTLY outperforms Naive confidence (p < 0.05)")
    else:
        logger.info(f"\n❌ RESULT: No significant improvement of Semantic over Naive (p ≥ 0.05 or negative effect)")

    return {
        'test': 'RQ2: Paired t-test (Semantic-Only vs Naive-Only, one-tailed)',
        'semantic_mean': float(np.mean(semantic)),
        'naive_mean': float(np.mean(naive)),
        'mean_improvement': mean_diff,
        'ci_95': ci_95,
        'std_improvement': float(np.std(diff, ddof=1)),
        't_statistic': float(t_stat),
        'p_value_raw': float(p_value),
        'cohen_d': float(cohen_d),
        'significant_raw': p_value < 0.05 and t_stat > 0,
        'domains': domains.tolist(),
        'semantic_aurocs': semantic.tolist(),
        'naive_aurocs': naive.tolist(),
    }


def chi_square_test_rq3a(features_dir):
    """
    RQ3a: Do hallucination rates differ significantly across domains?
    
    H0: Hallucination rates are equal across domains
    H1: Hallucination rates differ across domains
    """
    logger.info("\n" + "="*80)
    logger.info("RQ3a: HALLUCINATION RATE DIFFERENCES ACROSS DOMAINS")
    logger.info("="*80)
    
    domains = ['math', 'is_agents', 'psychology', 'medicine', 'finance']
    contingency_table = []
    domain_names = []
    
    for domain in domains:
        features_path = Path(features_dir) / f"{domain}_features.csv"
        
        if not features_path.exists():
            logger.warning(f"Features file not found for {domain}")
            continue
        
        df = pd.read_csv(features_path)
        
        # Count hallucinations vs faithful
        hall_count = df['hallucination_label'].sum()
        faithful_count = len(df) - hall_count
        
        contingency_table.append([hall_count, faithful_count])
        domain_names.append(domain)
        
        hall_rate = hall_count / len(df) * 100
        logger.info(f"  {domain:15s}: {hall_count:5d} hallucinations, {faithful_count:5d} faithful ({hall_rate:.1f}% hall. rate)")
    
    # Chi-square test
    contingency_array = np.array(contingency_table)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_array)

    # 95% CI for each domain's hallucination rate (Wilson score approximation)
    def proportion_ci(x, n, confidence=0.95):
        if n == 0:
            return 0.0, 0.0, 0.0
        p = x / n
        z = stats.norm.ppf((1 + confidence) / 2)
        se = np.sqrt(p * (1 - p) / n)
        margin = z * se
        low = max(0, p - margin)
        high = min(1, p + margin)
        return float(p), float(low), float(high)

    rate_cis = {}
    for i, d in enumerate(domain_names):
        x, n = contingency_table[i][0], contingency_table[i][0] + contingency_table[i][1]
        p, low, high = proportion_ci(x, n)
        rate_cis[d] = {'rate': p, 'ci_95': [low, high]}

    logger.info(f"\nChi-square Test Results:")
    logger.info(f"  Chi-square statistic: {chi2:.4f}")
    logger.info(f"  p-value:              {p_value:.6f}")
    logger.info(f"  Degrees of freedom:   {dof}")
    for d in domain_names:
        r = rate_cis[d]
        logger.info(f"  {d}: rate={r['rate']:.3f}, 95% CI=[{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}]")

    if p_value < 0.05:
        logger.info(f"\n✅ RESULT: Hallucination rates SIGNIFICANTLY differ across domains (p < 0.05)")
    else:
        logger.info(f"\n❌ RESULT: No significant difference in hallucination rates across domains (p ≥ 0.05)")

    return {
        'test': 'RQ3a: Chi-square test (hallucination rate differences)',
        'chi_square': float(chi2),
        'p_value_raw': float(p_value),
        'degrees_of_freedom': int(dof),
        'significant_raw': p_value < 0.05,
        'domains': domain_names,
        'contingency_table': contingency_array.tolist(),
        'rate_ci_per_domain': rate_cis,
    }


def feature_importance_variability_rq3c(ablation_dir):
    """
    RQ3c: Do features vary in importance across domains?
    
    Calculate coefficient of variation (CV) for each feature's importance
    across domains. High CV (> 0.3) indicates domain-specific feature.
    """
    logger.info("\n" + "="*80)
    logger.info("RQ3c: FEATURE IMPORTANCE VARIABILITY ACROSS DOMAINS")
    logger.info("="*80)
    
    domains = ['math', 'is_agents', 'psychology', 'medicine', 'finance']
    importance_data = {}
    
    # Load feature importance from each domain
    for domain in domains:
        importance_path = Path(ablation_dir) / f"{domain}_feature_importance.csv"
        
        if not importance_path.exists():
            logger.warning(f"Feature importance not found for {domain}")
            continue
        
        df = pd.read_csv(importance_path)
        
        for _, row in df.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if feature not in importance_data:
                importance_data[feature] = {}
            
            importance_data[feature][domain] = importance
    
    # Calculate variability metrics
    variability_results = []
    
    for feature, domain_importances in importance_data.items():
        if len(domain_importances) < 5:
            continue  # Need all 5 domains
        
        importances = [domain_importances[d] for d in domains if d in domain_importances]
        
        mean_imp = np.mean(importances)
        std_imp = np.std(importances, ddof=1)
        cv = std_imp / mean_imp if mean_imp > 0 else 0
        min_imp = np.min(importances)
        max_imp = np.max(importances)
        range_imp = max_imp - min_imp
        
        variability_results.append({
            'feature': feature,
            'mean_importance': mean_imp,
            'std_importance': std_imp,
            'coefficient_variation': cv,
            'min_importance': min_imp,
            'max_importance': max_imp,
            'importance_range': range_imp,
            'varies_across_domains': cv > 0.3
        })
    
    # Create dataframe and sort by CV
    variability_df = pd.DataFrame(variability_results)
    variability_df = variability_df.sort_values('coefficient_variation', ascending=False)
    
    # Count high-variance features
    high_var_count = (variability_df['coefficient_variation'] > 0.3).sum()
    low_var_count = (variability_df['coefficient_variation'] < 0.2).sum()
    
    logger.info(f"\nFeature Variability Summary:")
    logger.info(f"  Total features analyzed: {len(variability_df)}")
    logger.info(f"  High-variance features (CV > 0.3): {high_var_count}")
    logger.info(f"  Low-variance features (CV < 0.2): {low_var_count}")
    
    logger.info(f"\nTop 10 High-Variance Features (Domain-Specific):")
    for _, row in variability_df.head(10).iterrows():
        logger.info(f"  {row['feature']:30s}: CV={row['coefficient_variation']:.3f}, "
                   f"Range={row['importance_range']:.3f}")
    
    logger.info(f"\nTop 10 Low-Variance Features (Universal):")
    for _, row in variability_df.sort_values('coefficient_variation').head(10).iterrows():
        logger.info(f"  {row['feature']:30s}: CV={row['coefficient_variation']:.3f}, "
                   f"Mean={row['mean_importance']:.3f}")
    
    return variability_df, {
        'test': 'RQ3c: Feature importance variability analysis',
        'total_features': len(variability_df),
        'high_variance_count': high_var_count,
        'low_variance_count': low_var_count
    }


def convert_to_serializable(obj):
    """Convert numpy/types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (bool, np.bool_)):
        return int(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _run_rq1_rq2_with_metric(per_domain_df, metric_col='auroc_cv_mean'):
    """Run RQ1 and RQ2 using a given metric column (e.g. auroc_cv_mean instead of auroc)."""
    if metric_col not in per_domain_df.columns:
        return None, None
    df = per_domain_df.copy()
    df['auroc'] = df[metric_col].values
    rq1 = paired_t_test_rq1(df)
    rq2 = paired_t_test_rq2(df)
    return rq1, rq2


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run statistical tests for research questions"
    )
    parser.add_argument(
        '--ablation-dir',
        type=str,
        default='outputs/ablation',
        help='Directory containing ablation results'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default='data/features',
        help='Directory containing feature CSVs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/research_questions',
        help='Output directory for results'
    )

    args = parser.parse_args()

    try:
        logger.info("="*80)
        logger.info("STATISTICAL TESTS FOR RESEARCH QUESTIONS")
        logger.info("="*80)

        per_domain_path = Path(args.output_dir) / 'per_domain_ablation_breakdown.csv'
        if not per_domain_path.exists():
            logger.error(f"Per-domain breakdown not found: {per_domain_path}")
            logger.error("Run aggregate_ablation_results.py first!")
            return

        per_domain_df = pd.read_csv(per_domain_path)

        # --- Single-split tests ---
        results = {}
        results['rq1'] = paired_t_test_rq1(per_domain_df)
        results['rq2'] = paired_t_test_rq2(per_domain_df)
        results['rq3a'] = chi_square_test_rq3a(args.features_dir)

        variability_df, rq3c_summary = feature_importance_variability_rq3c(args.ablation_dir)
        results['rq3c'] = rq3c_summary

        # Multiple comparison correction (3 primary tests)
        p_values = [
            results['rq1']['p_value_raw'],
            results['rq2']['p_value_raw'],
            results['rq3a']['p_value_raw'],
        ]
        test_names = ['RQ1 (Hybrid vs Naive)', 'RQ2 (Semantic vs Naive)', 'RQ3a (Chi-square rates)']
        corrected_alpha, bonferroni_reject = bonferroni_correction(p_values, alpha=0.05)
        p_fdr, fdr_reject = fdr_correction(p_values, alpha=0.05)

        # Bonferroni adjusted p-value = min(1, p * n)
        p_bonferroni = [min(1.0, p * N_PRIMARY_TESTS) for p in p_values]

        results['rq1']['p_value_bonferroni'] = p_bonferroni[0]
        results['rq1']['p_value_fdr'] = p_fdr[0]
        results['rq1']['significant_bonferroni'] = bonferroni_reject[0]
        results['rq1']['significant_fdr'] = fdr_reject[0]
        results['rq2']['p_value_bonferroni'] = p_bonferroni[1]
        results['rq2']['p_value_fdr'] = p_fdr[1]
        results['rq2']['significant_bonferroni'] = bonferroni_reject[1]
        results['rq2']['significant_fdr'] = fdr_reject[1]
        results['rq3a']['p_value_bonferroni'] = p_bonferroni[2]
        results['rq3a']['p_value_fdr'] = p_fdr[2]
        results['rq3a']['significant_bonferroni'] = bonferroni_reject[2]
        results['rq3a']['significant_fdr'] = fdr_reject[2]

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Save multiple comparison corrections CSV
        mc_df = pd.DataFrame({
            'test': test_names,
            'p_value_raw': p_values,
            'p_value_bonferroni': p_bonferroni,
            'p_value_fdr': p_fdr,
            'significant_bonferroni': bonferroni_reject,
            'significant_fdr': fdr_reject,
            'bonferroni_alpha': corrected_alpha,
        })
        mc_path = Path(args.output_dir) / 'multiple_comparison_corrections.csv'
        mc_df.to_csv(mc_path, index=False)
        logger.info(f"✓ Multiple comparison corrections saved to {mc_path}")

        results_serializable = convert_to_serializable(results)
        summary_path = Path(args.output_dir) / 'statistical_tests_summary_single_split.json'
        with open(summary_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        logger.info(f"✓ Summary (single-split) saved to {summary_path}")

        # Backward compatibility
        with open(Path(args.output_dir) / 'statistical_tests_summary.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)

        variability_path = Path(args.output_dir) / 'rq3c_feature_importance_variability.csv'
        variability_df.to_csv(variability_path, index=False)
        logger.info(f"✓ Feature variability saved to {variability_path}")

        rq3a_df = pd.DataFrame({
            'domain': results['rq3a']['domains'],
            'hallucinations': [row[0] for row in results['rq3a']['contingency_table']],
            'faithful': [row[1] for row in results['rq3a']['contingency_table']]
        })
        rq3a_df['total'] = rq3a_df['hallucinations'] + rq3a_df['faithful']
        rq3a_df['hallucination_rate'] = rq3a_df['hallucinations'] / rq3a_df['total']
        rq3a_path = Path(args.output_dir) / 'rq3a_hallucination_rates.csv'
        rq3a_df.to_csv(rq3a_path, index=False)
        logger.info(f"✓ Hallucination rates saved to {rq3a_path}")

        # --- CV-based tests (if auroc_cv_mean present) ---
        if 'auroc_cv_mean' in per_domain_df.columns:
            logger.info("\n" + "="*80)
            logger.info("RUNNING CV-BASED STATISTICAL TESTS")
            logger.info("="*80)
            rq1_cv, rq2_cv = _run_rq1_rq2_with_metric(per_domain_df, 'auroc_cv_mean')
            if rq1_cv is not None and rq2_cv is not None:
                p_cv = [rq1_cv['p_value_raw'], rq2_cv['p_value_raw']]
                _, bonf_cv = bonferroni_correction(p_cv + [results['rq3a']['p_value_raw']], alpha=0.05)
                p_fdr_cv, fdr_cv = fdr_correction(p_cv + [results['rq3a']['p_value_raw']], alpha=0.05)
                rq1_cv['p_value_bonferroni'] = min(1.0, rq1_cv['p_value_raw'] * N_PRIMARY_TESTS)
                rq1_cv['p_value_fdr'] = p_fdr_cv[0]
                rq1_cv['significant_bonferroni'] = bonf_cv[0]
                rq1_cv['significant_fdr'] = fdr_cv[0]
                rq2_cv['p_value_bonferroni'] = min(1.0, rq2_cv['p_value_raw'] * N_PRIMARY_TESTS)
                rq2_cv['p_value_fdr'] = p_fdr_cv[1]
                rq2_cv['significant_bonferroni'] = bonf_cv[1]
                rq2_cv['significant_fdr'] = fdr_cv[1]
                results_cv = {
                    'rq1': rq1_cv,
                    'rq2': rq2_cv,
                    'rq3a': results['rq3a'],
                    'rq3c': results['rq3c'],
                }
                cv_path = Path(args.output_dir) / 'statistical_tests_summary_cv.json'
                with open(cv_path, 'w') as f:
                    json.dump(convert_to_serializable(results_cv), f, indent=2)
                logger.info(f"✓ CV-based summary saved to {cv_path}")
                comparison = {
                    'rq1_single_support': bool(results['rq1']['significant_raw']),
                    'rq1_cv_support': bool(rq1_cv['significant_raw']),
                    'rq1_single_p': float(results['rq1']['p_value_raw']),
                    'rq1_cv_p': float(rq1_cv['p_value_raw']),
                    'rq2_single_support': bool(results['rq2']['significant_raw']),
                    'rq2_cv_support': bool(rq2_cv['significant_raw']),
                    'rq2_single_p': float(results['rq2']['p_value_raw']),
                    'rq2_cv_p': float(rq2_cv['p_value_raw']),
                }
                comp_path = Path(args.output_dir) / 'statistical_tests_comparison.json'
                with open(comp_path, 'w') as f:
                    json.dump(convert_to_serializable(comparison), f, indent=2)
                logger.info(f"✓ Single vs CV comparison saved to {comp_path}")

        logger.info("\n" + "="*80)
        logger.info("ALL STATISTICAL TESTS COMPLETE")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Statistical tests failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
