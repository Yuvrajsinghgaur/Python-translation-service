"""
Configuration settings for the translation service
"""

import os
from typing import Dict, List
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Service configuration
    service_name: str = "NurseConnect AI Translation Service"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1
    
    # Model configuration
    model_cache_dir: str = "/app/model_cache"
    max_cache_size: int = 1000  # Maximum number of cached translations
    
    # Translation settings
    max_text_length: int = 5000
    default_confidence_threshold: float = 0.7
    batch_size: int = 32
    
    # Healthcare domain settings
    healthcare_terminology_enabled: bool = True
    medical_ner_enabled: bool = True
    
    # Supported languages
    supported_languages: List[str] = [
        "en", "de", "fr", "it", "es", "pt", "nl", "sv", "no", "da", 
        "fi", "pl", "cs", "hu", "ro", "bg", "hr", "sk", "sl", "et", 
        "lv", "lt", "mt", "ga", "cy", "eu", "ca", "gl","hi"    ]
    
    # Language pairs with specialized models
    specialized_pairs: Dict[str, List[str]] = {
        "healthcare": [
            "en-de", "en-fr", "en-it", "en-es", "en-pt",
            "de-en", "fr-en", "it-en", "es-en", "pt-en", "en-hi", "hi-en" 
        ]
    }
    
    # Model URLs and configurations
    model_configs: Dict[str, Dict] = {
        "marian_healthcare": {
            "base_url": "Helsinki-NLP/opus-mt-{source}-{target}",
            "domain": "healthcare",
            "max_length": 512
        },
        "mbart_multilingual": {
            "model_name": "facebook/mbart-large-50-many-to-many-mmt",
            "domain": "general",
            "max_length": 1024
        }
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()