"""
Configuration settings for the Hotel Agent System.
Contains API keys, thresholds, and other constants.
"""

from pydantic_settings import BaseSettings
from typing import Dict
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "443"))
    OPENSEARCH_USERNAME: str = os.getenv("OPENSEARCH_USERNAME", "")
    OPENSEARCH_PASSWORD: str = os.getenv("OPENSEARCH_PASSWORD", "")
    REGION: str = os.getenv("AWS_REGION", "")
    # Model Settings
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Thresholds
    CONFIDENCE_THRESHOLD: float = 0.7
    SIMILARITY_THRESHOLD: float = 0.8
    MAX_RETRIES: int = 3
    
    # RAG Settings
    CHUNK_SIZE: int = 10
    CHUNK_OVERLAP: int = 20
    MAX_RESULTS: int = 10
    
    # Memory Settings
    MEMORY_PERSISTENCE_PATH: Path = Path("data/memory.json")
    
    # Agent Prompts
    AGENT_PROMPTS: Dict[str, str] = {
        "dining": "You are a helpful hotel dining assistant. You can help with restaurant information, menus, and dining reservations.",
        "service": "You are a helpful hotel service assistant. You can help with housekeeping, room service, and maintenance requests."
    }
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "hotel_agent.log"
    
    class Config:
        env_file = ".env"

settings = Settings() 