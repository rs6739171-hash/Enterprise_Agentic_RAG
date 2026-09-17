import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT") or os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_rag")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY") or os.getenv("PORTKEY_API")
    GPT_SLUG = os.getenv("PORTKEY_PRIMARY_SLUG", "rag1")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID", "").strip()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2-preview")
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))


settings = Settings()


def validate_settings() -> None:
    required = {
        "QDRANT_CLUSTER_ENDPOINT": settings.QDRANT_URL,
        "QDRANT_API_KEY": settings.QDRANT_API_KEY,
        "PORTKEY_API_KEY": settings.PORTKEY_API_KEY,
    }
    if settings.EMBEDDING_BACKEND == "gemini":
        required["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    elif settings.EMBEDDING_BACKEND != "sentence-transformers":
        raise RuntimeError("EMBEDDING_BACKEND must be gemini or sentence-transformers.")
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
