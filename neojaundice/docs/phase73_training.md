# Phase 7.3 — Tracked, resumable training

## Purpose
This adds a **new** trainer (`scripts/train_tracked.py`) without modifying the validated Phase 7.2 baseline or overwriting E0/E1/E2 results. It trains on `train.csv`, selects the best model using `val.csv`, and **does not open `test.csv`**.

## Prerequisites
- Run from `/content/neojaundice/neojaundice` in a Colab **T4 GPU** runtime.
- Dataset extracted at `/content/neojaundice_data/NeoJaundice` and `data/processed/neojaundice/{train,val}.csv` available.
- `python -m pip install mlflow` for optional experiment tracking. Torch, torchvision, pandas, scikit-learn, Pillow and NumPy already required by Phase 7.2.

## Tests
```bash
python -m pytest tests/unit/test_train_tracked.py -q
python -m ruff check scripts/train_tracked.py tests/unit/test_train_tracked.py
```

## Initial GPU run
```bash
python -m scripts.train_tracked \
  --image-root /content/neojaundice_data/NeoJaundice \
  --task regression --preprocessing augment \
  --epochs 12 --patience 4 --batch-size 16 --workers 2 \
  --learning-rate 0.0001 --seed 42 \
  --output outputs/regression_e3_seed42 \
  --mlflow --mlflow-uri file:///content/neojaundice/neojaundice/mlruns
```

**This is a new experiment E3**, not a repeat of E0/E1/E2. If GPU time is scarce, skip the run until the persistence workflow below is ready.

## Output
- `best_model.pt`: model weights selected by validation loss, plus task/preprocessing metadata.
- `last_state.pt`: latest epoch, optimizer/scheduler states, RNG states, history, and configuration for resumption.
- `history.json`: epoch metrics, best epoch, and an explicit flag that the test set was not evaluated.
- `mlruns/`: MLflow tracking data, local to the Colab runtime unless copied to persistent storage.

## Persist to Drive **after every epoch**
Use a **persistent output directory on mounted Drive** for the first implementation, e.g.:
`--output /content/drive/MyDrive/NeoJaundice/checkpoints/regression_e3_seed42`
This avoids losing checkpoints when the runtime disconnects, but writes may be slower. For stronger reliability, use cloud object storage or periodically copy snapshots to Drive. For MLflow, set a persistent Drive-backed tracking URI (file-based MLflow storage on Drive can be slower).

## Resume
Run the **same command** with `--resume`; you may increase `--epochs` (total target epochs). The script checks key training configuration values and restores model, optimizer, scheduler, and RNG states. Only resume from your own trusted checkpoints: loading optimizer state uses pickle-backed PyTorch deserialization.

Note: exact bitwise reproducibility across interrupted GPU sessions is **not guaranteed**, particularly with multiprocessing workers and CUDA kernels. This is an engineering recovery mechanism, not proof of exact deterministic resumption.

## Clinical/research caveats
- `Treatment` is not validated as a medical diagnosis or treatment decision.
- Three images per patient: report patient-level results; no patient should cross splits.
- Avoid strong color transformations: color may be the signal of interest.
- Do not tune on the held-out test split.
- Dataset permissions and privacy controls must be checked before using third-party cloud services.
- Existing E2 (5 epochs, best epoch 5) should **not** be rerun merely to use this trainer. E3 tests longer training and monitoring as a distinct experiment.
