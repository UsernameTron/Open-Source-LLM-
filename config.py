import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Sentiment Analysis API"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8000",  # Local development
        "http://localhost:3000",  # React frontend (if used)
        # Add your production domains here
    ]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB in bytes
    ALLOWED_UPLOAD_TYPES: List[str] = [
        "application/pdf",
        "text/plain",
        "text/csv"
    ]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60  # Requests per minute
    
    # Security
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: str = os.getenv("API_KEY", "")  # Set this in production
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "4"))
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
