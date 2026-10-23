#!/usr/bin/env python3
"""Phase 7.5: run existing tracked classifier and record reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_label_semantics(project_root: Path) -> None:
    """Ensure the classification target is sourced from treatment_label."""
    dataset = project_root / "ml" / "training" / "dataset.py"
    if not dataset.is_file():
        raise FileNotFoundError(f"Missing dataset module: {dataset}")
    source = dataset.read_text(encoding="utf-8")
    if "treatment_label" not in source:
        raise RuntimeError(
            "Dataset implementation does not mention treatment_label. Verify the classification target before training."
        )


def build_command(
    args: argparse.Namespace, project_root: Path, output_dir: Path
) -> list[str]:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "train_tracked.py"),
        "--manifest-dir",
        str((project_root / args.manifest_dir).resolve()),
        "--image-root",
        str(Path(args.image_root).expanduser().resolve()),
        "--output",
        str(output_dir),
        "--task",
        "classification",
        "--preprocessing",
        args.preprocessing,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--mlflow",
        "--mlflow-uri",
        args.mlflow_uri,
        "--experiment",
        args.experiment,
    ]
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    if args.resume:
        cmd += ["--resume", str(Path(args.resume).expanduser().resolve())]
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".")
    p.add_argument("--manifest-dir", default="data/processed/manifests")
    p.add_argument("--image-root", required=True)
    p.add_argument(
        "--output-dir", default="artifacts/checkpoints/classification_treatment_seed42"
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--preprocessing", choices=("resize", "letterbox", "augment"), default="augment"
    )
    p.add_argument(
        "--mlflow-uri", default="sqlite:////content/neojaundice_mlflow/mlflow.db"
    )
    p.add_argument("--experiment", default="neojaundice-classification")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.workers < 0:
        p.error(
            "epochs, batch-size, patience must be positive; workers must be nonnegative"
        )
    if args.learning_rate <= 0:
        p.error("learning-rate must be positive")
    return args


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = (root / args.output_dir).resolve()
    trainer = root / "scripts" / "train_tracked.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"Missing existing trainer: {trainer}")
    verify_label_semantics(root)
    manifest_dir = (root / args.manifest_dir).resolve()
    if not manifest_dir.is_dir():
        raise FileNotFoundError(
            f"Missing manifest directory: {manifest_dir}; pass --manifest-dir"
        )
    image_root = Path(args.image_root).expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    output.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, root, output)
    print("Running:", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(cmd, cwd=root, check=False)
    record = {
        "phase": "7.5",
        "task": "classification",
        "label_column": "treatment_label",
        "label_semantics": "Unverified: do not interpret as clinical diagnosis or treatment recommendation",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": result.returncode,
        "command": cmd,
        "artifacts": {},
    }
    for name in ("best_model.pt", "last_state.pt", "history.json"):
        file = output / name
        if file.is_file():
            record["artifacts"][name] = {"path": str(file), "sha256": sha256_file(file)}
    record_file = output / "phase75_run_record.json"
    record_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Run record: {record_file}")
    if result.returncode:
        print("Training failed; inspect trainer logs and run record.", file=sys.stderr)
    else:
        print(
            "Classification training completed. Do not evaluate on test until model is frozen."
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
