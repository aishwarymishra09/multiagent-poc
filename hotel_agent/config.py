"""
Configuration settings for the Hotel Agent System.
Contains API keys, thresholds, and other constants.
"""

import streamlit as st
from pydantic_settings import BaseSettings
from typing import Dict
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = st.secrets["OPENAI_API_KEY"]
    OPENSEARCH_HOST: str = st.secrets["OPENSEARCH_HOST"]
    OPENSEARCH_PORT: int = int(st.secrets["OPENSEARCH_PORT"])
    OPENSEARCH_USERNAME: str = st.secrets["OPENSEARCH_USER"]
    OPENSEARCH_PASSWORD: str = st.secrets["OPENSEARCH_PASSWORD"]
    REGION: str = st.secrets["AWS_REGION"]
    # Model Settings
    MODEL_NAME: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Thresholds
    CONFIDENCE_THRESHOLD: float = 0.5
    SIMILARITY_THRESHOLD: float = 0.8
    MAX_RETRIES: int = 3
    
    # RAG Settings
    CHUNK_SIZE: int = 10
    CHUNK_OVERLAP: int = 20
    MAX_RESULTS: int = 10
    
    # Memory Settings
    MEMORY_PERSISTENCE_PATH: Path = Path("data/memory.json")
    
    # LangChain Configuration
    LANGCHAIN_TRACING_V2: bool = "true"
    LANGCHAIN_TRACING: bool = "true"
    
    LANGCHAIN_ENDPOINT: str = st.secrets["LANGCHAIN_ENDPOINT"]
    LANGCHAIN_API_KEY: str = st.secrets["LANGCHAIN_API_KEY"]
    LANGCHAIN_PROJECT: str = st.secrets["LANGCHAIN_PROJECT"]
    
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