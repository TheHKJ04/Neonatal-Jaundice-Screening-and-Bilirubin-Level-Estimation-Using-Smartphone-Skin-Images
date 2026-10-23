# Changelog entry — Phase 7.4

Add this entry to the project's root `CHANGELOG.md` after the evaluation script has been run successfully:

## [Unreleased] — Regression evaluation
- Added a held-out test evaluation script for the selected regression checkpoint.
- Added image-level and patient-level MAE/RMSE.
- Added a training-set-mean baseline, patient-level bootstrap confidence intervals, prediction CSVs, and diagnostic plots.
- Added checkpoint SHA-256 recording for artifact identity.
- Added tests for metric calculation, patient aggregation, and bootstrap reproducibility.
- Evaluation results remain pending until the script is run against the project's actual checkpoint and test split.
