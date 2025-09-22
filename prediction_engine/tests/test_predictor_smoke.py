"""Smoke tests for vendor error predictor."""

from app.core.prediction.services.predictor import VendorErrorPredictor


def test_predictor_returns_bounded_value():
    predictor = VendorErrorPredictor()
    adjustment = predictor.predict_adjustment(
        {
            "vendor_id": "odos",
            "amount_in_usd": 100.0,
            "output_amount_quote": "995",
        }
    )
    assert 0.0 <= adjustment <= 0.04
