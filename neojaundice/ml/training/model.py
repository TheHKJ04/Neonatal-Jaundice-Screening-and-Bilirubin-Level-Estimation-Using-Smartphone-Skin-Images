"""EfficientNet-B0 with one output for a single supervised task."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def calculate_loss(
    logits: torch.Tensor, target: torch.Tensor, task: str
) -> torch.Tensor:
    prediction = logits.flatten()
    if task == "regression":
        return nn.functional.smooth_l1_loss(prediction, target)
    if task == "classification":
        return nn.functional.binary_cross_entropy_with_logits(prediction, target)
    raise ValueError(f"Unknown task: {task}")
