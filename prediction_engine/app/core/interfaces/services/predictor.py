"""Predictor interface definitions."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class IPredictor(ABC):
    """Interface for predictor implementations."""

    @abstractmethod
    def predict_adjustment(self, features: Dict[str, Any]) -> float:
        """Return adjustment factor in [0, 0.04] for a single trade."""
        raise NotImplementedError
