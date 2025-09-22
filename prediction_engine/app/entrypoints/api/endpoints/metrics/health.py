"""API endpoint for health checks."""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.entrypoints.api.schemas.health import HealthStatusResponse

router = APIRouter(tags=["Metrics"])


@router.get(
    "",
    summary="API health check.",
    response_model=HealthStatusResponse,
)
async def health_check() -> HealthStatusResponse:
    """Returns the health status of the API."""

    return HealthStatusResponse(
        status="healthy", timestamp=datetime.now(timezone.utc).isoformat()
    )
