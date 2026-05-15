from functools import lru_cache
import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

IS_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT"))

if not IS_RAILWAY:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    IS_RAILWAY = bool(os.getenv("RAILWAY_ENVIRONMENT"))

API_PORT = int(os.getenv("PORT", "8001"))
BACKEND_URL = os.getenv("API_URL") or ("" if IS_RAILWAY else "http://127.0.0.1:8001/api/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL", "openai/gpt-4.1")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "btp_knowledge")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR") or ("/data/chroma_db" if IS_RAILWAY else "./chroma_db")
GMAIL_CREDENTIALS = os.getenv("GMAIL_CREDENTIALS", None)
GMAIL_TOKEN = os.getenv("GMAIL_TOKEN", None)
SECRET_KEY = os.getenv("SECRET_KEY") or os.getenv("RAILWAY_SERVICE_ID") or "dev-secret-change-me"


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = OPENAI_API_KEY
    openai_base_url: str = Field(
        default=OPENAI_BASE_URL,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "BASE_URL"),
    )

    # ChromaDB
    chroma_persist_dir: str = CHROMA_PERSIST_DIR
    chroma_collection_name: str = CHROMA_COLLECTION_NAME

    # Embedding model
    embedding_model: str = EMBEDDING_MODEL

    # LLM
    llm_model: str = Field(
        default=MODEL,
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL", "MODEL"),
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    retrieval_k: int = 6  # nombre de chunks retournés par recherche

    # Gmail API
    google_credentials_file: str = "credentials.json"
    google_token_file: str = "token.json"
    gmail_credentials: str | None = GMAIL_CREDENTIALS
    gmail_token: str | None = GMAIL_TOKEN
    google_redirect_uri: str = f"http://localhost:{API_PORT}/auth/gmail/callback"
    gmail_query: str = "newer_than:30d"
    gmail_max_results: int = 20

    # OCR / Vision
    gemini_api_key: str = ""
    vision_backend: str = "openai"
    vision_model: str = VISION_MODEL
    clip_model_name: str = "openai/clip-vit-base-patch32"
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    tesseract_cmd: str = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    ocr_pdf_text_threshold: int = 100
    ocr_image_text_threshold: int = 50

    # App
    app_env: str = "development"
    app_port: int = API_PORT
    backend_url: str = BACKEND_URL
    is_railway: bool = IS_RAILWAY
    secret_key: str = SECRET_KEY
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
