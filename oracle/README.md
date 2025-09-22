# Pricing Oracle API

HTTP API for ETH/USDC trades with ML-adjusted pricing.

## Setup

```bash
npm install
cp .env.example .env
```

Edit `.env` for your `ODOS_USER_ADDR` and `PREDICTOR_BASE_URL`.

## Run

```bash
npm run dev  # development
npm start    # production
```

Server runs on port 8080.

## Usage

**POST** `/api/price`

```bash
curl -X POST http://localhost:8080/api/price \
  -H "Content-Type: application/json" \
  -d '{
    "chain_id": 1,
    "input_token": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "output_token": "0x0000000000000000000000000000000000000000",
    "input_token_amount": "100000000"
  }'
```

Returns raw and ML-adjusted USDC-per-ETH price.

**GET** `/api/metrics/health` - Health check
