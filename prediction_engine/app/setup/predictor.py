"""This file contains the predictor dependency."""

from app.core.prediction.services.predictor import VendorErrorPredictor
from app.core.interfaces.services.predictor import IPredictor


def get_predictor() -> IPredictor:
    """Provide a predictor instance for dependency injection."""
    return VendorErrorPredictor()
