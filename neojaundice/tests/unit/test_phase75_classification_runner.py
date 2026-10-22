"""Tests for the Phase 7.5 wrapper."""
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'run_classification_phase75.py'
spec = importlib.util.spec_from_file_location('phase75', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sha256_file(tmp_path):
    f = tmp_path / 'data.bin'
    f.write_bytes(b'classification')
    assert module.sha256_file(f) == hashlib.sha256(b'classification').hexdigest()


def test_label_guard_rejects_missing_column(tmp_path):
    d = tmp_path / 'ml' / 'training'
    d.mkdir(parents=True)
    (d / 'dataset.py').write_text('target = 0', encoding='utf-8')
    with pytest.raises(RuntimeError, match='treatment_label'):
        module.verify_label_semantics(tmp_path)


def test_label_guard_accepts_declared_column(tmp_path):
    d = tmp_path / 'ml' / 'training'
    d.mkdir(parents=True)
    (d / 'dataset.py').write_text('target = row["treatment_label"]', encoding='utf-8')
    module.verify_label_semantics(tmp_path)


def test_command_uses_existing_trainer_and_classification(tmp_path):
    a = SimpleNamespace(manifest_dir='data/processed/manifests', image_root=str(tmp_path),
        preprocessing='augment', epochs=15, patience=4, batch_size=16,
        learning_rate=0.0001, seed=42, workers=2,
        mlflow_uri='sqlite:////tmp/mlflow.db', experiment='neojaundice-classification',
        no_pretrained=False, resume=None)
    cmd = module.build_command(a, tmp_path, tmp_path / 'output')
    assert '--task' in cmd and cmd[cmd.index('--task') + 1] == 'classification'
    assert '--mlflow' in cmd
    assert str(tmp_path / 'scripts' / 'train_tracked.py') in cmd
