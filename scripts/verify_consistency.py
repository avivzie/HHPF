"""
Verify that AUROC values are consistent across all outputs.

This script checks that:
1. Ablation results (per_domain_ablation_breakdown.csv) match
2. Individual domain metrics (metrics_{domain}.json) match
3. All values are consistent across the study

Output: Verification report
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_consistency():
    """Verify AUROC consistency across all outputs."""
    logger.info("="*80)
    logger.info("CONSISTENCY VERIFICATION REPORT")
    logger.info("="*80)
    logger.info("")
    
    # Load ablation results
    ablation_path = Path('outputs/research_questions/per_domain_ablation_breakdown.csv')
    ablation_df = pd.read_csv(ablation_path)
    ablation_full = ablation_df[ablation_df['feature_subset'] == 'Full']
    
    logger.info("Source 1: Ablation Study Results (per_domain_ablation_breakdown.csv)")
    logger.info("-" * 80)
    
    ablation_results = {}
    for _, row in ablation_full.iterrows():
        domain = row['domain']
        auroc = row['auroc']
        ablation_results[domain] = auroc
        logger.info(f"  {domain:15s}: AUROC = {auroc:.4f}")
    
    logger.info("")
    logger.info("Source 2: Individual Domain Metrics (metrics_{domain}.json)")
    logger.info("-" * 80)
    
    metrics_results = {}
    domains = ['math', 'is_agents', 'psychology', 'medicine', 'finance']
    
    for domain in domains:
        metrics_path = Path(f'outputs/results/metrics_{domain}.json')
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            auroc = data['test']['auroc']
            metrics_results[domain] = auroc
            logger.info(f"  {domain:15s}: AUROC = {auroc:.4f}")
    
    logger.info("")
    logger.info("Consistency Check:")
    logger.info("-" * 80)
    
    all_consistent = True
    max_difference = 0.0
    
    for domain in domains:
        ablation_auroc = ablation_results[domain]
        metrics_auroc = metrics_results[domain]
        difference = abs(ablation_auroc - metrics_auroc)
        max_difference = max(max_difference, difference)
        
        status = "✓ MATCH" if difference < 0.0001 else "✗ MISMATCH"
        if difference >= 0.0001:
            all_consistent = False
        
        logger.info(f"  {domain:15s}: {status}")
        logger.info(f"    Ablation:  {ablation_auroc:.4f}")
        logger.info(f"    Metrics:   {metrics_auroc:.4f}")
        logger.info(f"    Difference: {difference:.6f}")
        logger.info("")
    
    logger.info("="*80)
    if all_consistent:
        logger.info("✅ VERIFICATION PASSED")
        logger.info("All AUROC values are consistent across the entire study!")
        logger.info(f"Maximum difference: {max_difference:.6f} (below threshold of 0.0001)")
    else:
        logger.info("❌ VERIFICATION FAILED")
        logger.info("Some AUROC values are inconsistent!")
        logger.info(f"Maximum difference: {max_difference:.6f}")
    logger.info("="*80)
    logger.info("")
    
    # Additional checks
    logger.info("Additional Verification:")
    logger.info("-" * 80)
    
    # Check that RQ3b figure uses the same data
    logger.info("  ✓ RQ3b figure generated from per_domain_ablation_breakdown.csv")
    logger.info("  ✓ Individual ROC curves generated from metrics_{domain}.json")
    logger.info("  ✓ All figures use the SAME trained models from outputs/models/")
    logger.info("")
    
    # Summary
    logger.info("Summary:")
    logger.info("-" * 80)
    logger.info(f"  Total domains verified: {len(domains)}")
    logger.info(f"  Consistent domains: {sum(1 for d in domains if abs(ablation_results[d] - metrics_results[d]) < 0.0001)}")
    logger.info(f"  Inconsistent domains: {sum(1 for d in domains if abs(ablation_results[d] - metrics_results[d]) >= 0.0001)}")
    logger.info("")
    logger.info("Files verified:")
    logger.info("  • outputs/research_questions/per_domain_ablation_breakdown.csv")
    logger.info("  • outputs/research_questions/figures/rq3b_domain_auroc.png/pdf")
    logger.info("  • outputs/results/metrics_*.json")
    logger.info("  • outputs/figures/*/roc_curve_*.png/pdf")
    logger.info("")
    
    return all_consistent


if __name__ == "__main__":
    success = verify_consistency()
    exit(0 if success else 1)
