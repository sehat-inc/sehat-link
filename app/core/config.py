"""
Application Config File

This module handles configuration management, environment laoding, detection and parsing
- Ali Vijdaan
"""

import os
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Defining environment
class Environment(str, Enum):
    """
    Application environment types
    """

    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TEST = "test"


def get_environment() -> Environment:
    """
    Gets current environment
    """

    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "development" | "dev":
            return Environment.DEVELOPMENT
        case "test" | "testing":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


def load_env_file():
    """
    Load environment-specific .env file.
    """

    env = get_environment()
    print(f"Loading environment: {env}")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define env files in priority order
    env_files = [
        os.path.join(base_dir, f".env.{env.value}.local"),
        os.path.join(base_dir, f".env.{env.value}"),
        os.path.join(base_dir, ".env.local"),
        os.path.join(base_dir, ".env"),
    ]

    # Load the first env file that exists
    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(dotenv_path=env_file)
            print(f"Loaded environment from {env_file}")
            return env_file

    # Fall back to default if no env file found
    return None


ENV_FILE = load_env_file()


class Settings(BaseSettings):
    """
    Application Settings Managed via Pydantic
    """

    ENVIRONMENT: Environment = Field(default_factory=get_environment)
    PROJECT_NAME: str = "Sehat Link"
    PROJECT_VERSION: str = "0.0.1"
    DESCRIPTION: str = "AI Powered Healthcare Application"

    # LangGraph Configurations
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = "gemini-2.5-flash"

    # PostgreSQL Configurations
    POSTGRES_HOST: str = ""
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""

    # Logging Configurations
    LOG_DIR: Path = Path("logs")
    LOG_FORMAT: str = "json"
    LOG_LEVEL: str = "INFO"

    # Evaluation Configurations
    EVAL_DIR: Path = Path("evaluations")
    EVAL_LLM: str = ""
    EVAL_LLM_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("EVAL_LLM_API_KEY", mode="before")
    def default_api_key(cls, v, info):
        """
        Default evaluation API key to main LLM API key if not set
        """
        if not v:
            return info.get("LLM_API_KEY", "")
        return v
    
    def apply_environment_settings(self):
        """Apply environment-specific overrides dynamically."""
        overrides = {
            Environment.DEVELOPMENT: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG"
            },
            Environment.PRODUCTION: {
                "DEBUG": False,
                "LOG_LEVEL": "WARNING"
            },
            Environment.TEST: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG"
            }
        }

        env_overrides = overrides.get(self.ENVIRONMENT, {})
        for key, value in env_overrides.items():
            if key not in os.environ:
                setattr(self, key, value)


settings = Settings()
settings.apply_environment_settings()

