"""This file contains global application settings."""

from os import path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

DOTENV_FILE = ".env" if path.isfile(".env") else None


class Settings(BaseSettings):
    """Application settings."""

    # Application
    environment: str = "local"
    application_name: str = "prediction_engine"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = ConfigDict(env_file=DOTENV_FILE)


settings = Settings()
