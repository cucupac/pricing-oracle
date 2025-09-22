"""API setup module."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.setup.event_loop import get_event_loop
from app.entrypoints.api.endpoints.metrics import health
from app.entrypoints.api.endpoints.internal import prediction


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """Context manager for the application's lifespan."""
    await get_event_loop()
    yield


def create_app() -> FastAPI:
    """Creates the FastAPI application."""
    fastapi_app = FastAPI(
        title="The Error PredictedErrorResponse API",
        description="Makes predicted pricing errors available via API call.",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    fastapi_app.include_router(health.router, prefix="/health")
    fastapi_app.include_router(prediction.router, prefix="/predict")

    # CORS (Cross-Origin Resource Sharing)
    origins = ["*"]
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return fastapi_app


app = create_app()


def entry() -> None:
    """Starts the prediction engine."""

    uvicorn.run(
        "app.entrypoints.api.setup:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    entry()
