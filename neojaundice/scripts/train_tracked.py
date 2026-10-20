"""Resumable EfficientNet baseline with optional MLflow tracking (train/val only)."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.training.dataset import NeoJaundiceDataset
from ml.training.model import build_model, calculate_loss
from scripts.train_baseline import evaluate, set_seed


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/neojaundice")
    )
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--task", choices=("regression", "classification"), default="regression"
    )
    p.add_argument(
        "--preprocessing", choices=("resize", "letterbox", "augment"), default="augment"
    )
    p.add_argument(
        "--epochs", type=int, default=12, help="Total epochs, including resumed epochs"
    )
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--exclude-zero", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--mlflow", action="store_true", help="Track run in local MLflow file store"
    )
    p.add_argument(
        "--mlflow-uri", default=None, help="Example: file:///content/.../mlruns"
    )
    p.add_argument("--experiment", default="neojaundice-regression")
    return p


def validate(a: argparse.Namespace) -> None:
    if a.epochs < 1 or a.patience < 1 or a.batch_size < 1 or a.workers < 0:
        raise ValueError("epochs, patience, batch-size must be positive; workers >= 0")
    if not np.isfinite(a.learning_rate) or a.learning_rate <= 0:
        raise ValueError("learning-rate must be positive and finite")
    if not np.isfinite(a.min_delta) or a.min_delta < 0:
        raise ValueError("min-delta must be nonnegative and finite")
    if a.resume and not (a.output / "last_state.pt").is_file():
        raise FileNotFoundError("--resume requested but last_state.pt not found")
    if not a.resume and (a.output / "last_state.pt").exists():
        raise FileExistsError(
            "Output already contains training state; use --resume or a new output"
        )


def configuration(a: argparse.Namespace) -> dict:
    return {
        k: getattr(a, k)
        for k in (
            "task",
            "preprocessing",
            "batch_size",
            "learning_rate",
            "seed",
            "no_pretrained",
            "exclude_zero",
        )
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def restore_rng(state: dict) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def capture_rng() -> dict:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def train(a: argparse.Namespace) -> dict:
    validate(a)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.output.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: NeoJaundiceDataset(
            a.manifest_dir / f"{split}.csv",
            a.image_root,
            a.task,
            a.preprocessing,
            training=split == "train",
            exclude_zero=a.exclude_zero,
        )
        for split in ("train", "val")
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=a.batch_size,
            shuffle=True,
            num_workers=a.workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=a.batch_size,
            shuffle=False,
            num_workers=a.workers,
        ),
    }
    model = build_model(pretrained=not a.no_pretrained and not a.resume).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    history: list[dict] = []
    best_loss, best_epoch, bad_epochs, start_epoch = float("inf"), 0, 0, 1
    if a.resume:
        # Only load checkpoints generated by this project and stored in trusted storage.
        state = torch.load(
            a.output / "last_state.pt", map_location="cpu", weights_only=False
        )
        if state["config"] != configuration(a):
            raise ValueError(
                "Resume configuration differs from checkpoint (except epochs/patience)"
            )
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        # Optimizer tensors loaded on CPU must be moved to the selected training device.
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        history = state["history"]
        best_loss, best_epoch, bad_epochs = (
            state["best_loss"],
            state["best_epoch"],
            state["bad_epochs"],
        )
        start_epoch = state["epoch"] + 1
        restore_rng(state["rng"])
    mlflow = None
    if a.mlflow:
        import mlflow as mlflow_module

        mlflow = mlflow_module
        tracking_uri = (
            a.mlflow_uri
            or os.environ.get("MLFLOW_TRACKING_URI")
            or f"sqlite:///{(Path.cwd() / 'mlflow.db').as_posix()}"
        )
        if tracking_uri.startswith("file:"):
            raise ValueError(
                "Legacy MLflow file tracking is not supported by "
                "the current project configuration. "
                "Use a SQLite tracking URI instead."
            )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(a.experiment)
        mlflow.start_run(run_name=a.output.name)
        mlflow.log_params(
            {
                **configuration(a),
                "epochs_requested": a.epochs,
                "patience": a.patience,
            }
        )
    try:
        for epoch in range(start_epoch, a.epochs + 1):
            model.train()
            loss_sum = 0.0
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch["image"].to(device))
                loss = calculate_loss(predictions, batch["target"].to(device), a.task)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(batch["target"])
            val_loss, metrics = evaluate(model, loaders["val"], a.task, device)
            scheduler.step(val_loss)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / len(datasets["train"]),
                "val_loss": val_loss,
                "val_metrics": metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            improved = val_loss < best_loss - a.min_delta
            if improved:
                best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "task": a.task,
                        "preprocessing": a.preprocessing,
                        "epoch": epoch,
                        "validation_loss": val_loss,
                        "val_metrics": metrics,
                    },
                    a.output / "best_model.pt",
                )
            else:
                bad_epochs += 1
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": configuration(a),
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "rng": capture_rng(),
            }
            atomic_torch_save(state, a.output / "last_state.pt")
            summary = {
                "config": {
                    **configuration(a),
                    "epochs": a.epochs,
                    "patience": a.patience,
                },
                "device": str(device),
                "train_images": len(datasets["train"]),
                "val_images": len(datasets["val"]),
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "test_set_evaluated": False,
            }
            atomic_json_save(summary, a.output / "history.json")
            if mlflow:
                log = {
                    "train_loss": row["train_loss"],
                    "val_loss": val_loss,
                    "lr": row["learning_rate"],
                }
                for level, values in metrics.items():
                    log.update(
                        {
                            f"val_{level}_{key}": value
                            for key, value in values.items()
                            if isinstance(value, (int, float))
                        }
                    )
                mlflow.log_metrics(log, step=epoch)
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "best_epoch": best_epoch,
                        "val_metrics": metrics,
                    }
                ),
                flush=True,
            )
            if bad_epochs >= a.patience:
                print(f"Early stopping after epoch {epoch}; best epoch {best_epoch}")
                break
        if mlflow:
            mlflow.log_metric("best_val_loss", best_loss)
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_artifact(str(a.output / "history.json"))
        return summary if history else {"history": [], "best_epoch": best_epoch}
    finally:
        if mlflow:
            mlflow.end_run()


def main() -> int:
    train(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
