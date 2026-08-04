# config.py - Configuration and environment setup

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path="new.env")

class Config:
    """Application configuration"""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    CLASSIFIER_MODEL = "llama-3.1-8b-instant"
    EXPLANATION_MODEL = "gpt-4o-mini"
    QUIZ_MODEL = "llama-3.1-8b-instant"

    CHROMA_PERSIST_DIR = "./chroma_db"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    MAX_QUESTIONS_PER_SESSION = 20

    @classmethod
    def validate_config(cls):
        required_keys = ["GROQ_API_KEY", "OPENROUTER_API_KEY"]
        for key in required_keys:
            if not getattr(cls, key):
                raise ValueError(f"Missing required environment variable: {key}")
