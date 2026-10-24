"""Benchmark NeoJaundice real-model inference latency.

Usage:
    python scripts/benchmark_inference.py path/to/test_image.jpg
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from backend.app.core.config import get_settings
from backend.app.services.model_registry import ModelRegistry
from ml.training.dataset import make_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NeoJaundice inference latency."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to a local JPEG, PNG, or WebP test image.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of warm inference repetitions (default: 20).",
    )

    return parser.parse_args()


def load_image_tensor(image_path: Path) -> torch.Tensor:
    """Load an image and apply the project's inference preprocessing."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Use a JPEG, PNG, or WebP test image.")

        image = ImageOps.exif_transpose(image).convert("RGB")

        transform = make_transform(
            mode="resize",
            training=False,
        )

        return transform(image).unsqueeze(0)


def percentile(values: list[float], percentage: float) -> float:
    """Calculate a percentile using a nearest-rank method."""

    ordered = sorted(values)

    index = max(
        0,
        min(
            len(ordered) - 1,
            int((percentage / 100.0) * len(ordered) + 0.999999) - 1,
        ),
    )

    return ordered[index]


def main() -> int:
    args = parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    image_tensor = load_image_tensor(args.image)

    settings = get_settings()

    registry = ModelRegistry(
        regression_checkpoint=settings.regression_checkpoint,
        classification_checkpoint=settings.classification_checkpoint,
    )

    # First call includes lazy model loading.
    start = time.perf_counter()

    with torch.inference_mode():
        prediction = registry.predict(image_tensor)

    cold_start_ms = (time.perf_counter() - start) * 1000.0

    # Warm-up after the models have loaded.
    with torch.inference_mode():
        registry.predict(image_tensor)

    warm_latencies_ms: list[float] = []

    for _ in range(args.repeats):
        start = time.perf_counter()

        with torch.inference_mode():
            registry.predict(image_tensor)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        warm_latencies_ms.append(elapsed_ms)

    report = {
        "benchmark": "neojaundice_inference",
        "device": "cpu",
        "checkpoint_loading_included_in_cold_start": True,
        "image_dimensions": list(image_tensor.shape),
        "warm_repetitions": args.repeats,
        "cold_start_latency_ms": round(cold_start_ms, 3),
        "warm_latency_ms": {
            "mean": round(statistics.mean(warm_latencies_ms), 3),
            "median": round(statistics.median(warm_latencies_ms), 3),
            "p95": round(percentile(warm_latencies_ms, 95), 3),
            "minimum": round(min(warm_latencies_ms), 3),
            "maximum": round(max(warm_latencies_ms), 3),
        },
        "prediction": prediction,
        "clinical_use": "research_only",
    }

    output_dir = Path("reports") / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "inference_benchmark.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nBenchmark report saved to: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
