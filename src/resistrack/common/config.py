"""Application configuration using Pydantic Settings."""

from typing import Literal

from pydantic_settings import BaseSettings


class ResisTrackConfig(BaseSettings):
    """Application configuration loaded from environment variables."""

    aws_region: str = "us-east-1"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    sagemaker_endpoint_name: str = ""
    healthlake_datastore_id: str = ""
    rds_secret_arn: str = ""

    model_config = {"env_prefix": "RESISTRACK_", "case_sensitive": False}


__all__ = ["ResisTrackConfig"]
