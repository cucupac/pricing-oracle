# Prediction Engine

ML service that predicts DEX quote errors. Send a quote, get an adjustment factor.

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Place model files in `app/core/prediction/weights/`.

## Run

```bash
uvicorn app.entrypoints.api.setup:app --port 8000 --reload
```

## API

**POST** `/predict`

Input: DEX quote data  
Output: `{"adjustment_factor": 0.9909739446640015, "model_version": "v1.0"}`

Example request:

```bash
curl -X POST http://localhost:8000/predict \\
  -H 'Content-Type: application/json' \\
  -d '{
    "vendor_id": "odos",
    "chain_id": 1,
    "input_token": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "output_token": "0x0000000000000000000000000000000000000000",
    "input_amount": "10000000",
    "output_amount_quote": "250000000000000",
    "amount_in_usd": 10.0,
    "quote_gas_usd": 0.10,
    "hop_count": 2,
    "min_pool_tvl_usd": 1000000.0
  }'
```

## Test

```bash
pytest
```
