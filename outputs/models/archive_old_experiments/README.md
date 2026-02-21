# Archived Experimental Models

**Archive Date:** February 14, 2026  
**Reason:** These are old experimental models from calibration testing (Feb 7, 2026)

## Archived Models

### Calibration Experiments (Feb 7)
- `xgboost_finance_isotonic.pkl` - Finance with isotonic calibration
- `xgboost_finance_platt.pkl` - Finance with Platt scaling calibration
- `xgboost_psychology_isotonic.pkl` - Psychology with isotonic calibration
- `xgboost_psychology_platt.pkl` - Psychology with Platt scaling calibration
- `xgboost_psychology_baseline_backup.pkl` - Psychology backup from earlier run

## Final Models (Still in Parent Directory)

The 5 final models used in the thesis are in `outputs/models/`:

| Model | Date | AUROC | Status |
|-------|------|-------|--------|
| xgboost_math.pkl | Feb 14 16:52 | 0.7973 | ✅ Final |
| xgboost_is_agents.pkl | Feb 14 16:52 | 0.7027 | ✅ Final |
| xgboost_psychology.pkl | Feb 14 16:52 | 0.6715 | ✅ Final |
| xgboost_medicine.pkl | Feb 14 16:53 | 0.6192 | ✅ Final |
| xgboost_finance.pkl | Feb 14 16:53 | 0.6320 | ✅ Final |

## Why These Were Archived

These models were from earlier calibration experiments where we tested:
- **Isotonic calibration** - Non-parametric calibration method
- **Platt scaling** - Logistic regression calibration

The final thesis uses models WITHOUT additional calibration, as the raw XGBoost probabilities provided good calibration (especially for Medicine with ECE=0.057).

## Can These Be Deleted?

Yes, these can be safely deleted if needed. They are not used in any thesis results or figures.

**Keep only for:** Historical reference of calibration experiments
