
import os

from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

MAX_ISSUES = 5


@lru_cache(maxsize=1)
def get_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from .env"
        )

    return Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=2
    )
