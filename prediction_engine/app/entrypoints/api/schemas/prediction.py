"""Schemas for prediction API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, constr

HexAddr = constr(pattern=r"^0x[a-fA-F0-9]{40}$")


class PredictionRequest(BaseModel):
    """The JSON request body for the prediction endpoint."""

    vendor_id: Literal["odos", "1inch"]
    chain_id: int = Field(ge=1)
    input_token: str = Field(min_length=42, max_length=42, pattern=r"^0x")
    output_token: str = Field(min_length=42, max_length=42, pattern=r"^0x")
    input_amount: str
    output_amount_quote: str
    amount_in_usd: Optional[float] = None
    quote_gas_usd: Optional[float] = None
    hop_count: Optional[float] = None
    min_pool_tvl_usd: Optional[float] = None


class PredictionResponse(BaseModel):
    """The response model for the prediction endpoint."""

    adjustment_factor: float
    model_version: str
