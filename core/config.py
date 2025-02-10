"""Configuration settings for the LLM Engine."""
from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    WORKERS: int = 4
    TIMEOUT: int = 60
    KEEP_ALIVE: int = 300
    MAX_CONNECTIONS: int = 100
    BACKLOG: int = 128
    
    # Monitoring settings
    PROMETHEUS_PORT: int = 9090
    ERROR_LOG_PATH: str = "logs/errors.log"
    ACCESS_LOG_PATH: str = "logs/access.log"
    MAX_STORED_ERRORS: int = 1000
    ERROR_REFRESH_INTERVAL: int = 2  # seconds
    METRICS_REFRESH_INTERVAL: int = 5  # seconds
    
    # Model settings
    MAX_BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512
    MODEL_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
