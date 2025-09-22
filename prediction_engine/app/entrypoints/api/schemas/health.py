"""Pydantic models for health-related API responses."""

from pydantic import BaseModel, Field


class HealthStatusResponse(BaseModel):
    """API response model for the health API endpoint."""

    status: str = Field(
        description="The status of the API.",
        examples=["healthy"],
    )
    timestamp: str = Field(
        description="The current date and time as an ISO formated string.",
        examples=["2025-09-20T12:34:56.789012"],
    )
