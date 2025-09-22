"""Utilities for loading predictor artifacts."""

import json
import os
from typing import Any, Dict

import torch

from . import BoundedMultiLayerPerceptron

ROOT = "app/core/prediction"
WEIGHTS_DIRECTORY = f"{ROOT}/weights"
ARTIFACTS_DIRECTORY = f"{ROOT}/artifacts"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIRECTORY, "vendor_error_mlp.pt")
META_PATH = os.path.join(ARTIFACTS_DIRECTORY, "vendor_error_meta.json")


def load_meta() -> Dict[str, Any]:
    """Load metadata accompanying the trained model."""
    with open(META_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model_from_meta(meta: Dict[str, Any]) -> BoundedMultiLayerPerceptron:
    """Instantiate and hydrate the trained model using metadata."""
    input_dim = len(meta["feature_cols"])
    hidden = meta.get("hidden_sizes", [32, 16])
    dropout = float(meta.get("dropout", 0.10))
    model = BoundedMultiLayerPerceptron(input_dim, hidden, dropout_rate=dropout)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
