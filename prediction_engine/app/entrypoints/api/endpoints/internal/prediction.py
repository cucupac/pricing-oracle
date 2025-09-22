"""Prediction endpoint exposing vendor error adjustments."""

from fastapi import APIRouter, Depends

from app.core.interfaces.services.predictor import IPredictor
from app.entrypoints.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.setup.predictor import get_predictor

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest, predictor: IPredictor = Depends(get_predictor)
) -> PredictionResponse:
    """Predict the adjustment factor for a quoted trade."""

    adjustment = predictor.predict_adjustment(request.model_dump())
    adjustment_factor = max(0.0, min(1.0, 1.0 - adjustment))

    return PredictionResponse(
        adjustment_factor=adjustment_factor,
        model_version="vendor-error-mlp.pt",
    )
