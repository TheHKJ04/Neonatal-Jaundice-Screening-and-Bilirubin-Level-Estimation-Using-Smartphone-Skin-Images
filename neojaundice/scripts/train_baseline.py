"""Train a reproducible exploratory EfficientNet-B0 baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.training.dataset import NeoJaundiceDataset
from ml.training.metrics import summarize_predictions
from ml.training.model import build_model, calculate_loss


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/processed/neojaundice")
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/external/neojaundice/extracted/NeoJaundice"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/efficientnet_baseline")
    )
    parser.add_argument(
        "--task", choices=["regression", "classification"], required=True
    )
    parser.add_argument(
        "--preprocessing", choices=["resize", "letterbox", "augment"], default="resize"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--exclude-zero", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, task, device):
    model.eval()
    records = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target = batch["target"].to(device)
            logits = model(images)
            loss = calculate_loss(logits, target, task)
            total_loss += loss.item() * len(target)
            values = logits.flatten()
            if task == "classification":
                values = torch.sigmoid(values)
            for pid, iid, y, p in zip(
                batch["patient_id"],
                batch["image_id"],
                target.cpu().tolist(),
                values.cpu().tolist(),
            ):
                records.append(
                    {"patient_id": pid, "image_id": iid, "target": y, "prediction": p}
                )
    return total_loss / len(loader.dataset), summarize_predictions(records, task)


def main() -> int:
    args = arguments()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.workers < 0
    ):
        raise ValueError("Invalid training hyperparameters")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {
        name: NeoJaundiceDataset(
            args.manifest_dir / f"{name}.csv",
            args.image_root,
            args.task,
            args.preprocessing,
            training=name == "train",
            exclude_zero=args.exclude_zero,
        )
        for name in ("train", "val")
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss = calculate_loss(logits, batch["target"].to(device), args.task)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch["target"])
        val_loss, metrics = evaluate(model, val_loader, args.task, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / len(datasets["train"]),
            "val_loss": val_loss,
            "val_metrics": metrics,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task": args.task,
                    "preprocessing": args.preprocessing,
                    "epoch": epoch,
                    "validation_loss": val_loss,
                },
                args.output / "best_model.pt",
            )
    (args.output / "history.json").write_text(
        json.dumps(
            {
                "config": {
                    "task": args.task,
                    "preprocessing": args.preprocessing,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "pretrained": not args.no_pretrained,
                    "exclude_zero": args.exclude_zero,
                },
                "device": str(device),
                "train_images": len(datasets["train"]),
                "val_images": len(datasets["val"]),
                "history": history,
                "test_set_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best checkpoint and history to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
