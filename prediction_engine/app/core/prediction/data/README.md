# Synthetic Trade Dataset Generation

This folder contains a Python script to generate a **synthetic dataset** for the **Price Oracle Estimation System** MVP. The output CSV is used to train and test _vendor-error–first_ adjustments for **ETH/USDC swaps** on Ethereum.

## Output

-   **File:** `trades.csv`
-   **Rows:** 100,000 synthetic trade records
-   **Columns:** 18 fields covering trade parameters, quotes, execution results, and route characteristics

```
chain, vendor, in_token, out_token, amount_in_token, amount_in_usd,
quote_out_token, quote_out_usd, quote_gas_usd, spot_price_in,
spot_price_out, gas_price_wei, actual_out_token, actual_out_usd,
actual_gas_usd, predicted_adjustment_factor, hop_count, min_pool_tvl_usd
```

## Generation Rules

### Trade Configuration

-   **Chain:** `ethereum` (fixed)
-   **Direction:** USDC → ETH (_selling USDC to receive ETH_)
-   **Input Token:** USDC (`0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48`)
-   **Output Token:** ETH (`0x0000000000000000000000000000000000000000`)
-   **Vendors:** `{"odos", "1inch"}` (_uniform random selection_)

### Trade Sizes

**Input amounts** are sampled from discrete size tiers:

-   **Token amounts:** `{1M, 10M, 100M, 1B, 10B, 100B, 1T}` USDC base units _(6 decimals)_
-   **USD equivalents:** `{$1, $10, $100, $1K, $10K, $100K, $1M}`

### Quote Generation

The quote generation follows a **uniform token sampling** approach to ensure realistic price variation:

1. **Sample token-out uniformly** over the range implied by ETH ∈ **[4000, 4200]**:

    - `token_range_low = (amount_in_token/1e6) / 4200 * 1e18`
    - `token_range_high = (amount_in_token/1e6) / 4000 * 1e18`

2. **Compute implied ETH price** for internal consistency:

    - `eth_price_used = amount_in_usd / (quote_out_token / 1e18)`
    - _Clamped to [4000, 4200] for numerical safety_

3. **Set derived fields:**
    - `quote_out_usd = (quote_out_token / 1e18) * eth_price_used`
    - `spot_price_in = 1` _(USDC is pegged to $1)_
    - `spot_price_out = eth_price_used` _(uses same implied price)_

### Gas Parameters

-   **Quote gas cost:** `Uniform[2, 5]` USD
-   **Actual gas cost:** `= quote_gas_usd` _(no gas execution variance)_
-   **Gas price:** `5e9` wei _(fixed)_

### Route Characteristics

-   **Hop count:** `{1, 2, 3, 4}` _(uniform random)_
-   **Minimum pool TVL:** `Uniform[100, 1_000_000]` USD

## Execution Error Model

_Negative Slippage Only_

The system models **realistic execution degradation** where actual output is _always ≤ quoted output_:

### Error Components

The execution error `error_frac` is **half-normal negative**, combining two additive penalty factors:

1. **Hop penalty:** Up to **-2%** based on route complexity

    - `hop_component = 0.02 * (hop_count - 1) / 3`

2. **TVL penalty:** Up to **-2%** based on liquidity depth

    - `tvl_component = 0.02 * (1 - (tvl - 100) / (1_000_000 - 100))`

3. **Combined sampling:**
    - `max_reduction = hop_component + tvl_component` _(≤ 4% total)_
    - Sample `z ~ |Normal(0, σ = max_reduction/2)|`
    - `error_frac = -min(z, max_reduction)` _(always ≤ 0)_

### Final Calculations

-   **Actual output:** `actual_out_token = quote_out_token * (1 + error_frac)`
-   **Actual USD value:** `actual_out_usd = (actual_out_token / 1e18) * eth_price_used`
-   **Adjustment factor:** `1 - (actual_out_token / quote_out_token)` _(always ≥ 0)_

## Technical Notes

-   **Token precision:** Large token amounts (wei values) are stored as **stringified integers** to prevent scientific notation in CSV output
-   **Decimal arithmetic:** Uses Python's `Decimal` module for high-precision calculations
-   **Deterministic randomness:** Each run produces different data while maintaining statistical properties

## Usage

```bash
python app/core/prediction/data/generate_trades.py
```

**Output location:** `app/core/prediction/data/trades.csv`
