"""End-to-end tests for the prediction API."""

from fastapi.testclient import TestClient

from app.entrypoints.api.setup import create_app


def test_prediction_endpoint_returns_high_adjustment_factor() -> None:
    client = TestClient(create_app())
    payload = {
        "vendor_id": "odos",
        "chain_id": 1,
        "input_token": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "output_token": "0x0000000000000000000000000000000000000000",
        "input_amount": "10000000",
        "output_amount_quote": "250000000000000",
        "amount_in_usd": 10.0,
        "quote_gas_usd": 0.10,
        "hop_count": 2,
        "min_pool_tvl_usd": 1_000_000.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["adjustment_factor"] >= 0.96
