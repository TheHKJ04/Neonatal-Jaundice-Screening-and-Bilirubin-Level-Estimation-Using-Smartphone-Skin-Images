#!/usr/bin/env python3
"""Evaluate a frozen NeoJaundice binary classifier on the held-out test set.

This script reports dataset-label prediction performance only. It does not establish
clinical validity for the source `treatment_label` field.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, classification_report)
from torch.utils.data import DataLoader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binary_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError('y_true and probabilities must have the same nonzero length')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    result = {
        'count': int(len(y)),
        'positive_count': int(y.sum()),
        'negative_count': int(len(y) - y.sum()),
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall_sensitivity': float(recall_score(y, pred, zero_division=0)),
        'specificity': specificity,
        'f1': float(f1_score(y, pred, zero_division=0)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        'brier_score': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }
    return result


def bootstrap_patient_ci(patient_df: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Percentile bootstrap over independent patients, not correlated image rows."""
    if patient_df.empty or patient_df['target'].nunique() < 2:
        return {'method': 'patient-level percentile bootstrap', 'replicates': 0,
                'note': 'Not computed: patient table is empty or contains only one class.'}
    rng = np.random.default_rng(seed)
    rows = patient_df.reset_index(drop=True)
    metrics: dict[str, list[float]] = {k: [] for k in ('accuracy', 'precision', 'recall_sensitivity', 'specificity', 'f1', 'roc_auc')}
    n = len(rows)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = rows.iloc[idx]
        y = sample['target'].to_numpy(dtype=int)
        p = sample['probability'].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        m = binary_metrics(y, p, threshold=0.5)
        for key in metrics:
            val = m.get(key)
            if val is not None and np.isfinite(val):
                metrics[key].append(float(val))
    intervals = {}
    for key, values in metrics.items():
        intervals[key] = ({'lower_95': float(np.percentile(values, 2.5)),
                           'upper_95': float(np.percentile(values, 97.5)),
                           'valid_replicates': len(values)} if values else None)
    return {'method': 'patient-level percentile bootstrap; threshold fixed at 0.5',
            'requested_replicates': n_bootstrap, 'seed': seed, 'intervals': intervals}


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older torch versions
        obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f'Could not locate a state dictionary in {checkpoint_path}')
    # Handle DataParallel checkpoints.
    if all(str(k).startswith('module.') for k in obj):
        obj = {str(k)[7:]: v for k, v in obj.items()}
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', default='.')
    p.add_argument('--manifest-dir', default='data/processed/manifests')
    p.add_argument('--test-manifest', default=None, help='Explicit test CSV path; otherwise auto-detect common names.')
    p.add_argument('--image-root', required=True)
    p.add_argument('--checkpoint', default='artifacts/checkpoints/classification_treatment_seed42/best_model.pt')
    p.add_argument('--output-dir', default='artifacts/evaluation/classification')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threshold', type=float, default=0.5, help='Must be fixed using validation only before test evaluation.')
    p.add_argument('--bootstrap-replicates', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--preprocessing', choices=('resize', 'letterbox'), default='resize')
    return p.parse_args()


def resolve_test_manifest(root: Path, manifest_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f'Test manifest not found: {path}')
        return path.resolve()
    candidates = [manifest_dir / name for name in (
        'test.csv', 'test_manifest.csv', 'test_manifest.parquet.csv', 'test_split.csv')]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('Could not auto-detect test manifest. Pass --test-manifest /path/to/test.csv. Checked: ' + ', '.join(map(str, candidates)))


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser()
    if not manifest_dir.is_absolute():
        manifest_dir = root / manifest_dir
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    image_root = Path(args.image_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    if not image_root.is_dir():
        raise FileNotFoundError(f'Image root not found: {image_root}')
    if not 0 < args.threshold < 1:
        raise ValueError('--threshold must be between 0 and 1')
    test_manifest = resolve_test_manifest(root, manifest_dir, args.test_manifest)
    frame = pd.read_csv(test_manifest)
    required = {'patient_id', 'image_id', 'treatment_label'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Test manifest missing required columns: {missing}. Columns found: {list(frame.columns)}')
    labels = pd.to_numeric(frame['treatment_label'], errors='coerce')
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError('treatment_label must contain only binary values 0 and 1')
    frame['treatment_label'] = labels.astype(int)
    # Keep test data untouched: this script never trains or selects a threshold.
    sys.path.insert(0, str(root))
    from ml.training.dataset import NeoJaundiceDataset
    from ml.training.model import build_model

    dataset = NeoJaundiceDataset(frame, image_root=str(image_root), task='classification',
                                 preprocessing=args.preprocessing, training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(pretrained=False).to(device)
    state = load_state_dict(checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    probs, targets, patient_ids, image_ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            # Dataset returns a dictionary containing image, target and IDs.
            images = batch["image"]
            target = batch["target"]
            patients = batch["patient_id"]
            images_id = batch["image_id"]
            logits = model(images.to(device)).reshape(-1)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.extend(batch_probs.tolist())
            targets.extend(torch.as_tensor(target).reshape(-1).cpu().numpy().astype(int).tolist())
            patient_ids.extend([str(x) for x in patients])
            image_ids.extend([str(x) for x in images_id])
    if len(probs) != len(frame):
        raise RuntimeError(f'Dataset yielded {len(probs)} records but manifest has {len(frame)} rows')
    predictions = pd.DataFrame({'image_id': image_ids, 'patient_id': patient_ids,
                                'target': targets, 'probability': probs})
    predictions['prediction'] = (predictions['probability'] >= args.threshold).astype(int)
    image_metrics = binary_metrics(predictions['target'], predictions['probability'], args.threshold)
    # Patient-level target must be consistent across a patient's image rows.
    patient_rows = []
    inconsistent = []
    for pid, group in predictions.groupby('patient_id', sort=True):
        target_values = group['target'].unique()
        if len(target_values) != 1:
            inconsistent.append(str(pid))
            continue
        patient_rows.append({'patient_id': str(pid), 'target': int(target_values[0]),
                             'probability': float(group['probability'].mean()),
                             'image_count': int(len(group))})
    patient_df = pd.DataFrame(patient_rows)
    if patient_df.empty:
        raise ValueError('No patients had consistent labels; cannot compute patient-level metrics')
    patient_metrics = binary_metrics(patient_df['target'], patient_df['probability'], args.threshold)
    patient_ci = bootstrap_patient_ci(patient_df, args.bootstrap_replicates, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
    patient_df['prediction'] = (patient_df['probability'] >= args.threshold).astype(int)
    patient_df.to_csv(output_dir / 'patient_predictions.csv', index=False)
    report = {
        'phase': '7.6', 'task': 'binary dataset-label classification',
        'target_column': 'treatment_label',
        'target_semantics': 'Unverified; results must not be interpreted as clinical diagnosis or treatment recommendation.',
        'test_manifest': str(test_manifest), 'test_rows': int(len(frame)),
        'unique_patients_in_manifest': int(frame['patient_id'].nunique()),
        'patients_evaluated': int(len(patient_df)), 'patients_excluded_for_inconsistent_labels': inconsistent,
        'threshold': float(args.threshold), 'threshold_policy': 'Fixed before test evaluation; do not tune on test set.',
        'device': str(device), 'preprocessing': args.preprocessing,
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256_file(checkpoint),
        'image_level': image_metrics, 'patient_level': patient_metrics,
        'patient_level_bootstrap_95ci': patient_ci,
        'classification_report_image_level': classification_report(
            predictions['target'], predictions['prediction'], labels=[0, 1],
            output_dict=True, zero_division=0),
        'output_dir': str(output_dir.resolve()),
    }
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    md = [
        '# NeoJaundice Phase 7.6 — Classification Test Evaluation', '',
        '> **Research-only result.** `treatment_label` semantics have not been verified. Do not interpret these metrics as clinical screening accuracy or treatment guidance.', '',
        f'- Test manifest: `{test_manifest}`', f'- Test image rows: {len(frame)}',
        f'- Patients evaluated: {len(patient_df)}', f'- Fixed threshold: `{args.threshold:.3f}`',
        f'- Checkpoint SHA-256: `{report["checkpoint_sha256"]}`', '',
        '## Image-level metrics', '', '```json', json.dumps(image_metrics, indent=2), '```', '',
        '## Patient-level metrics', '', '```json', json.dumps(patient_metrics, indent=2), '```', '',
        '## Patient-level bootstrap confidence intervals', '', '```json', json.dumps(patient_ci, indent=2), '```', '',
        '## Interpretation limits', '',
        '- Patient-level results aggregate image probabilities by mean; labels must be consistent within each patient.',
        '- The threshold is fixed at 0.5; it must not be tuned using test outcomes.',
        '- Confidence intervals use a percentile bootstrap over patients, not images.',
        '- Performance against an unverified dataset label does not establish clinical validity.', '',
    ]
    (output_dir / 'evaluation_report.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({'image_level': image_metrics, 'patient_level': patient_metrics,
                      'patient_level_bootstrap_95ci': patient_ci,
                      'output_dir': str(output_dir.resolve())}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
