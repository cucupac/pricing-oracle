"""Generate synthetic trade data for the Price Oracle Estimation System MVP.

Output:
  - estimator/app/core/prediction/data/trades.csv

Columns:
  chain, vendor, in_token, out_token, amount_in_token, amount_in_usd,
  quote_out_token, quote_out_usd, quote_gas_usd, spot_price_in,
  spot_price_out, gas_price_wei, actual_out_token, actual_out_usd,
  actual_gas_usd, predicted_adjustment_factor, hop_count, min_pool_tvl_usd

Key features:
  - Quote generation samples UNIFORMLY over the token-out range implied by ETH∈[4000,4200]
  - The ETH price used for quote fields is the IMPLIED price from the sampled token-out
  - Negative-only execution error based on hop count and pool TVL
  - Token amounts returned as stringified integers to avoid scientific notation
  - 100,000 rows of synthetic USDC→ETH trade data
"""

import os
import random
from decimal import Decimal, getcontext, ROUND_FLOOR
from typing import Dict, Any, Tuple, List
import pandas as pd

# High precision for wei math
getcontext().prec = 50

# ----------------------------
# Constants
# ----------------------------
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
ETH_ADDRESS = "0x0000000000000000000000000000000000000000"

TRADE_AMOUNTS_USDC = [1e6, 10e6, 100e6, 1_000e6, 10_000e6, 100_000e6, 1_000_000e6]
TRADE_AMOUNTS_USD = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]

SUPPORTED_VENDORS = ["odos", "1inch"]

ETH_PRICE_MIN = Decimal("4000")
ETH_PRICE_MAX = Decimal("4200")

GAS_COST_MIN_USD = 2.0
GAS_COST_MAX_USD = 5.0
GAS_PRICE_WEI = 5_000_000_000  # fits int
USDC_SPOT_PRICE = 1.0

MIN_POOL_TVL_USD = 100.0
MAX_POOL_TVL_USD = 1_000_000.0
MIN_HOP_COUNT = 1
MAX_HOP_COUNT = 4

DATASET_ROWS = 100_000
WEI_PER_ETH = Decimal(10) ** 18  # exact integer wei


def select_random_trade_amount() -> Tuple[float, float]:
    """Return a randomly selected USDC token amount and its USD equivalent."""
    index = random.randrange(len(TRADE_AMOUNTS_USDC))
    return TRADE_AMOUNTS_USDC[index], TRADE_AMOUNTS_USD[index]


def select_random_vendor() -> str:
    """Return a randomly selected vendor from the supported list."""
    return random.choice(SUPPORTED_VENDORS)


def sample_hop_count() -> int:
    """Return a random hop count between configured bounds."""
    return random.randint(MIN_HOP_COUNT, MAX_HOP_COUNT)


def sample_min_pool_tvl_usd() -> float:
    """Return a random minimum pool TVL in USD within configured bounds."""
    return random.uniform(MIN_POOL_TVL_USD, MAX_POOL_TVL_USD)


def sample_quote_gas_usd() -> float:
    """Return a random quoted gas cost in USD within configured bounds."""
    return random.uniform(GAS_COST_MIN_USD, GAS_COST_MAX_USD)


def compute_quote_out_uniform_token(
    amount_in_token: float, amount_in_usd: float
) -> Tuple[str, float, float]:
    """Sample token-out uniformly from ETH∈[4000,4200] range and compute implied price, returning token-out as stringified integer wei."""
    amount_token_decimal = Decimal(str(amount_in_token))
    amount_usd_decimal = Decimal(str(amount_in_usd))

    # Range for ETH-out in wei implied by price bounds
    low_token_wei = (
        (amount_token_decimal / Decimal("1e6")) / ETH_PRICE_MAX * WEI_PER_ETH
    )
    high_token_wei = (
        (amount_token_decimal / Decimal("1e6")) / ETH_PRICE_MIN * WEI_PER_ETH
    )
    min_wei, max_wei = (
        (low_token_wei, high_token_wei)
        if low_token_wei <= high_token_wei
        else (high_token_wei, low_token_wei)
    )

    # Uniform sample in Decimal, then floor to integer wei
    random_factor = Decimal(str(random.random()))
    sampled_wei = min_wei + (max_wei - min_wei) * random_factor
    quote_out_token_wei = int(sampled_wei.to_integral_value(rounding=ROUND_FLOOR))

    # Implied ETH price consistent with sampled wei
    implied_eth_price = amount_usd_decimal / (
        Decimal(quote_out_token_wei) / WEI_PER_ETH
    )
    if implied_eth_price < ETH_PRICE_MIN:
        implied_eth_price = ETH_PRICE_MIN
    elif implied_eth_price > ETH_PRICE_MAX:
        implied_eth_price = ETH_PRICE_MAX

    quote_out_usd = float(
        (Decimal(quote_out_token_wei) / WEI_PER_ETH) * implied_eth_price
    )
    return str(quote_out_token_wei), quote_out_usd, float(implied_eth_price)


def sample_negative_error_fraction(hop_count: int, min_pool_tvl_usd: float) -> float:
    """Return a negative-only slippage fraction based on hop count and pool TVL."""
    hop_penalty_component = Decimal("0.02") * Decimal(hop_count - 1) / Decimal("3")
    tvl_normalized = (
        Decimal(str(min_pool_tvl_usd)) - Decimal(str(MIN_POOL_TVL_USD))
    ) / (Decimal(str(MAX_POOL_TVL_USD)) - Decimal(str(MIN_POOL_TVL_USD)))
    tvl_penalty_component = Decimal("0.02") * (Decimal("1") - tvl_normalized)
    max_penalty = hop_penalty_component + tvl_penalty_component

    if max_penalty <= 0:
        return 0.0

    noise_sigma = float(max_penalty) / 2.0
    half_normal_sample = abs(random.gauss(0.0, noise_sigma))
    return -min(half_normal_sample, float(max_penalty))


def generate_trade_row() -> Dict[str, Any]:
    """Generate a single row of synthetic trade data following the specified schema."""
    amount_in_token, amount_in_usd = select_random_trade_amount()
    vendor = select_random_vendor()
    hop_count = sample_hop_count()
    min_pool_tvl_usd = sample_min_pool_tvl_usd()

    quote_out_token_str, quote_out_usd, eth_price_used = (
        compute_quote_out_uniform_token(amount_in_token, amount_in_usd)
    )
    quote_gas_usd = sample_quote_gas_usd()

    # Apply negative slippage to integer wei and floor to integer wei
    execution_error_fraction = sample_negative_error_fraction(
        hop_count, min_pool_tvl_usd
    )
    quoted_wei = Decimal(quote_out_token_str)
    actual_wei = (
        quoted_wei * (Decimal("1") + Decimal(str(execution_error_fraction)))
    ).to_integral_value(rounding=ROUND_FLOOR)
    if actual_wei < 0:
        actual_wei = Decimal(0)

    actual_out_token_str = str(int(actual_wei))
    actual_out_usd = float((actual_wei / WEI_PER_ETH) * Decimal(str(eth_price_used)))

    # Predicted adjustment factor from exact integers
    predicted_adjustment_factor = (
        float(Decimal("1") - (actual_wei / quoted_wei)) if quoted_wei > 0 else 0.0
    )

    return {
        "chain": "ethereum",
        "vendor": vendor,
        "in_token": USDC_ADDRESS,
        "out_token": ETH_ADDRESS,
        "amount_in_token": amount_in_token,
        "amount_in_usd": amount_in_usd,
        "quote_out_token": quote_out_token_str,  # stringified integer wei (no sci notation)
        "quote_out_usd": quote_out_usd,
        "quote_gas_usd": quote_gas_usd,
        "spot_price_in": USDC_SPOT_PRICE,
        "spot_price_out": eth_price_used,
        "gas_price_wei": GAS_PRICE_WEI,
        "actual_out_token": actual_out_token_str,  # stringified integer wei (no sci notation)
        "actual_out_usd": actual_out_usd,
        "actual_gas_usd": quote_gas_usd,
        "predicted_adjustment_factor": predicted_adjustment_factor,
        "hop_count": hop_count,
        "min_pool_tvl_usd": min_pool_tvl_usd,
    }


def create_synthetic_dataset(number_of_rows: int) -> pd.DataFrame:
    """Generate a complete dataset with the specified number of synthetic trade rows."""
    trade_rows: List[Dict[str, Any]] = [
        generate_trade_row() for _ in range(number_of_rows)
    ]
    dataset = pd.DataFrame(trade_rows)

    # Maintain exact column ordering from specification
    column_order = [
        "chain",
        "vendor",
        "in_token",
        "out_token",
        "amount_in_token",
        "amount_in_usd",
        "quote_out_token",
        "quote_out_usd",
        "quote_gas_usd",
        "spot_price_in",
        "spot_price_out",
        "gas_price_wei",
        "actual_out_token",
        "actual_out_usd",
        "actual_gas_usd",
        "predicted_adjustment_factor",
        "hop_count",
        "min_pool_tvl_usd",
    ]

    return dataset[column_order]


def main() -> None:
    """Generate synthetic trade dataset and save to CSV file."""
    output_file_path = os.path.join(os.path.dirname(__file__), "trades.csv")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    synthetic_dataset = create_synthetic_dataset(DATASET_ROWS)
    synthetic_dataset.to_csv(output_file_path, index=False)

    print(f"Wrote {len(synthetic_dataset):,} rows to {output_file_path}")


if __name__ == "__main__":
    main()
