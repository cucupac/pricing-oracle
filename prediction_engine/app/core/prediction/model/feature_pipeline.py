"""Feature derivation and vectorization helpers."""

from typing import Any, Dict, List
import math
from decimal import Decimal, InvalidOperation
import numpy as np

REQUIRED_FEATURE_KEYS = [
    "vendor_onehot_odos",
    "vendor_onehot_1inch",
    "log_amount_in_usd",
    "log_output_amount_quote",
    "quote_gas_usd",
    "hop_count",
    "log_min_pool_tvl_usd",
    "log_amount_x_hops",
    "odos_x_log_amount",
    "inch_x_log_amount",
    "odos_x_hops",
    "inch_x_hops",
]


def _log1p_safe(value: float) -> float:
    return math.log1p(max(value, 0.0))


def derive_features(payload: Dict[str, Any]) -> Dict[str, float]:
    """Build the feature dictionary from incoming request payload."""

    vendor = payload.get("vendor_id", "odos")
    onehot_odos = 1.0 if vendor == "odos" else 0.0
    onehot_1inch = 1.0 if vendor == "1inch" else 0.0

    amount_in_usd = float(payload.get("amount_in_usd", 0.0))
    raw_output_amount = payload.get("output_amount_quote")
    output_amount_quote = 0.0
    if raw_output_amount is not None:
        try:
            output_amount_quote = float(Decimal(str(raw_output_amount)))
        except (InvalidOperation, ValueError):
            output_amount_quote = 0.0

    feature_dict: Dict[str, float] = {
        "vendor_onehot_odos": onehot_odos,
        "vendor_onehot_1inch": onehot_1inch,
        "log_amount_in_usd": _log1p_safe(amount_in_usd),
        "log_output_amount_quote": _log1p_safe(output_amount_quote),
    }

    quote_gas_usd = payload.get("quote_gas_usd")
    hop_count = payload.get("hop_count")
    min_pool_tvl_usd = payload.get("min_pool_tvl_usd")

    if quote_gas_usd is not None:
        feature_dict["quote_gas_usd"] = float(quote_gas_usd)
    if hop_count is not None:
        feature_dict["hop_count"] = float(hop_count)
    if min_pool_tvl_usd is not None:
        feature_dict["log_min_pool_tvl_usd"] = _log1p_safe(float(min_pool_tvl_usd))

    if "log_amount_in_usd" in feature_dict and "hop_count" in feature_dict:
        feature_dict["log_amount_x_hops"] = (
            feature_dict["log_amount_in_usd"] * feature_dict["hop_count"]
        )
    if "log_amount_in_usd" in feature_dict:
        feature_dict["odos_x_log_amount"] = (
            onehot_odos * feature_dict["log_amount_in_usd"]
        )
        feature_dict["inch_x_log_amount"] = (
            onehot_1inch * feature_dict["log_amount_in_usd"]
        )
    if "hop_count" in feature_dict:
        feature_dict["odos_x_hops"] = onehot_odos * feature_dict["hop_count"]
        feature_dict["inch_x_hops"] = onehot_1inch * feature_dict["hop_count"]

    return feature_dict


def vectorize_in_order(
    feature_dict: Dict[str, float],
    feature_cols: List[str],
    scaler_mean: List[float],
    scaler_std: List[float],
) -> np.ndarray:
    """Produce standardized feature vector in the specified order."""

    vector = np.zeros(len(feature_cols), dtype=np.float32)
    for idx, column in enumerate(feature_cols):
        mean_value = float(scaler_mean[idx])
        std_value = float(scaler_std[idx]) or 1e-6
        value = feature_dict.get(column, mean_value)
        vector[idx] = (float(value) - mean_value) / std_value
    return vector.reshape(1, -1)
