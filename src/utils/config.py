import os
from typing import Optional

class Config:
    """Configuration management"""
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    
    # API Configuration
    API_BASE_URL: str = os.getenv('API_BASE_URL', 'https://api.example.com')
    API_KEY: Optional[str] = os.getenv('API_KEY')
    
    # Database Configuration
    DATABASE_URL: Optional[str] = os.getenv('DATABASE_URL')
    
    # AWS Configuration
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # Model Configuration
    MODEL_BUCKET: str = os.getenv('MODEL_BUCKET', 'ml-models')
    DEFAULT_MODEL_NAME: str = os.getenv('DEFAULT_MODEL_NAME', 'default')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
