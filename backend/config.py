"""
Central configuration. All paths and tunables live here.
Copy .env.example → .env and fill in ANTHROPIC_API_KEY.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # ── Anthropic ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    vision_model: str = "claude-sonnet-4-6"    # for image captioning

    # ── Embedding model (local sentence-transformers) ──────────────────────────
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_text_collection: str = "manual_text"
    qdrant_image_collection: str = "manual_images"

    # ── SQLite ─────────────────────────────────────────────────────────────────
    sqlite_path: Path = ROOT / "knowledge" / "tables.db"

    # ── Knowledge store paths ──────────────────────────────────────────────────
    knowledge_dir: Path = ROOT / "knowledge"
    images_dir: Path = ROOT / "knowledge" / "images"
    screenshots_dir: Path = ROOT / "knowledge" / "screenshots"
    manuals_dir: Path = ROOT / "manuals" / "pdf"

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 512          # tokens (approximate; we use char ÷ 4)
    chunk_overlap: int = 64
    min_chunk_chars: int = 80      # drop tiny noise chunks

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_text: int = 6
    top_k_images: int = 3

    # ── Agent ─────────────────────────────────────────────────────────────────
    max_agent_iterations: int = 8
    agent_temperature: float = 0.2


settings = Settings()
