import hashlib
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = [
    "outputs/regression_e0/best_model.pt",
    "outputs/regression_e1/best_model.pt",
    "outputs/regression_e2/best_model.pt",
    "outputs/smoke_regression/best_model.pt",
    "outputs/classification_e0/best_model.pt",
    "artifacts/transferred/regression_best_model.pt",
    "artifacts/transferred/classification_best_model.pt",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_checkpoint(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path

    print("\n" + "=" * 75)
    print(f"CHECKPOINT: {relative_path}")
    print("=" * 75)

    if not path.is_file():
        print("STATUS: NOT FOUND")
        return

    print(f"Size: {path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"SHA256: {sha256_file(path)}")

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        print("STATUS: SAFE LOADING FAILED")
        print(f"Error: {exc}")
        print("Do not bypass safe loading for an untrusted file.")
        return

    if not isinstance(checkpoint, dict):
        print(f"Unexpected checkpoint type: {type(checkpoint)}")
        return

    print("\nMetadata:")

    for key, value in checkpoint.items():
        if key in ("model_state_dict", "state_dict", "model"):
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value).__name__}")

    state_dict = None

    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            state_dict = candidate
            print(f"\nWeights key: {key}")
            break

    if state_dict is None:
        print("\nNo recognized model state dictionary found.")
        print("Available keys:", list(checkpoint.keys()))
        return

    tensor_count = sum(isinstance(value, torch.Tensor) for value in state_dict.values())

    print(f"State dictionary entries: {len(state_dict)}")
    print(f"Tensor entries: {tensor_count}")

    print("\nFirst 12 parameter entries:")

    for name, value in list(state_dict.items())[:12]:
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {tuple(value.shape)}")
        else:
            print(f"  {name}: {type(value).__name__}")


def main() -> None:
    print("NEOJAUNDICE - CHECKPOINT AUDIT")
    print("Project:", PROJECT_ROOT)

    for checkpoint in CHECKPOINTS:
        inspect_checkpoint(checkpoint)


if __name__ == "__main__":
    main()
