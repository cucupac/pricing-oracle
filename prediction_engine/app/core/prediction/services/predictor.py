"""Concrete predictor implementation for vendor error adjustments."""

from typing import Any, Dict

import numpy as np
import torch

from app.core.interfaces.services.predictor import IPredictor
from app.core.prediction.model.feature_pipeline import (
    derive_features,
    vectorize_in_order,
)
from app.core.prediction.model.load import (
    build_model_from_meta,
    load_meta,
)

MAX_ADJUSTMENT = 0.04


class VendorErrorPredictor(IPredictor):
    """Predictor that loads a trained MLP and serves adjustment factors."""

    def __init__(self) -> None:
        self.meta = load_meta()
        self.model = build_model_from_meta(self.meta)
        scaler = self.meta["scaler"]
        self._feature_cols = self.meta["feature_cols"]
        self._mean = scaler["mean"]
        self._std = scaler["std"]

    def predict_adjustment(self, features: Dict[str, Any]) -> float:
        derived = derive_features(features)
        feature_vector = vectorize_in_order(
            derived, self._feature_cols, self._mean, self._std
        )
        tensor = torch.from_numpy(feature_vector.astype(np.float32))
        with torch.no_grad():
            scaled = float(self.model(tensor).item())
        adjustment = max(0.0, min(MAX_ADJUSTMENT, scaled * MAX_ADJUSTMENT))
        return adjustment
