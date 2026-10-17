# Phase 7.2 — EfficientNet-B0 supervised baseline

Uses the verified NeoJaundice patient-disjoint split manifests. No test data is loaded by the training script. Validation loss selects checkpoints. Reports both image-level and patient-aggregated exploratory metrics.

Tasks: blood bilirubin regression (mg/dL, modality not independently verified), or classification of the dataset's `Treatment` label (not a clinical recommendation).

Preprocessing modes: `resize` (E0), `letterbox` (E1), and `augment` (E2; mild geometry only). No color jitter. Pretrained ImageNet weights are downloaded by torchvision on first use. `--no-pretrained` is for smoke tests only.

Review zero-bilirubin labels and source measurement definitions before presenting regression results. `--exclude-zero` is a documented sensitivity analysis, not a default silent deletion. Repeated images per infant are correlated. Patient-level metrics aggregate image predictions, and clinical validation remains false.

Outputs: `best_model.pt` and `history.json` in the chosen output directory. Store checkpoints outside Git and version approved artifacts later with MLflow/DVC. Save the dataset and split versions before final evaluation.

Next: smoke test, real training, inspect overfitting, then add MLflow and controlled E1/E2 comparisons. Never use the test split for tuning.
