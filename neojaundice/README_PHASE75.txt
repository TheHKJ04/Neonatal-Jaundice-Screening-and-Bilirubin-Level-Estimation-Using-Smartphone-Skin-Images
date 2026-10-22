NeoJaundice Phase 7.5 classification patch

Copy the directories scripts/, tests/, docs/ into the NeoJaundice repository root.
This patch adds new files and does not overwrite the existing trainer.

1. Confirm the classification target maps to treatment_label in ml/training/dataset.py.
2. Confirm the manifest directory (default data/processed/manifests).
3. Run python -m pytest tests/unit/test_phase75_classification_runner.py -v
4. Run python scripts/run_classification_phase75.py --image-root /content/neojaundice_data/NeoJaundice --dry-run
5. Run again without --dry-run to train.

This patch does NOT contain datasets or model weights and does NOT claim that classification training has been executed.
