import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")
    RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    PROMPT_CACHE_RETENTION = "24h"

    REVIEWS_PARQUET = BASE_DIR / "data" / "embedded_1k_reviews.parquet"
    WW2_PARQUET = BASE_DIR / "data" / "embedded_ww2_chunks.parquet"
    INSTRUCTIONS_PATH = BASE_DIR / "instructions" / "prompt.txt"

    MAX_TOOL_LOOP_ITERATIONS = 5
    TOP_K = 3
    MAX_CONTENT_LENGTH = 1_000_000

    DEFAULT_POET_PROMPT = "Write a one-sentence story about a robot."
    DEFAULT_INSTRUCTIONS_QUESTION = "How would I declare a variable for a last name?"
    DEFAULT_CALENDAR_TEXT = "Alice and Bob are going to a science fair on Friday."
    DEFAULT_MATH_TEXT = "how can I solve 8x + 7 = -23"


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    OPENAI_API_KEY = "test-key"


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
