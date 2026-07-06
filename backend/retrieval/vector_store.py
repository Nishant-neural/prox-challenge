"""
Qdrant Vector Store
===================
Two collections:
  • manual_text   — chunked text passages, embedded with sentence-transformers
  • manual_images — image captions + keywords, embedded the same way

Provides:
  • upsert_chunks(chunks)
  • upsert_images(images)
  • search_text(query, top_k) → list[SearchHit]
  • search_images(query, top_k) → list[SearchHit]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.preprocessing.pdf_extractor import TextChunk, ImageRecord

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Search hit
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]

    @property
    def page(self) -> int:
        return self.payload.get("page", 0)

    @property
    def content(self) -> str:
        return self.payload.get("content", self.payload.get("description", ""))

    @property
    def section(self) -> str:
        return self.payload.get("section", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Store
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Wrapper around Qdrant for the OmniPro manual.
    Uses a local sentence-transformer model so no external embedding API is needed.
    """

    def __init__(self):
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        console.print("[cyan]Loading embedding model…[/cyan]")
        self._model = SentenceTransformer(settings.embed_model)
        self._ensure_collections()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _ensure_collections(self):
        existing = {c.name for c in self._client.get_collections().collections}

        for name in [settings.qdrant_text_collection, settings.qdrant_image_collection]:
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=settings.embed_dim,
                        distance=Distance.COSINE,
                    ),
                )
                console.print(f"[green]✓[/green] Created Qdrant collection: {name}")
            else:
                console.print(f"[dim]Collection already exists: {name}[/dim]")

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    # ── Upsert text chunks ─────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[TextChunk], batch_size: int = 64):
        """Embed and upsert text chunks into manual_text collection."""
        console.print(f"[cyan]Upserting {len(chunks)} text chunks…[/cyan]")

        for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting text"):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            vectors = self._embed(texts)

            points = [
                PointStruct(
                    id=_str_to_uint64(c.chunk_id),
                    vector=vec,
                    payload={
                        "chunk_id": c.chunk_id,
                        "page": c.page,
                        "section": c.section,
                        "content": c.content,
                        "source_pdf": c.source_pdf,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                    },
                )
                for c, vec in zip(batch, vectors)
            ]
            self._client.upsert(
                collection_name=settings.qdrant_text_collection,
                points=points,
            )

        console.print(f"[green]✓[/green] {len(chunks)} chunks upserted to '{settings.qdrant_text_collection}'")

    # ── Upsert image records ───────────────────────────────────────────────────

    def upsert_images(self, images: list[ImageRecord], batch_size: int = 32):
        """Embed image captions + keywords and upsert into manual_images collection."""
        console.print(f"[cyan]Upserting {len(images)} image records…[/cyan]")

        for i in tqdm(range(0, len(images), batch_size), desc="Upserting images"):
            batch = images[i : i + batch_size]
            # Embed a rich text combining caption, description, and keywords
            texts = [
                f"{img.caption}. {img.description}. Keywords: {', '.join(img.keywords)}"
                for img in batch
            ]
            vectors = self._embed(texts)

            points = [
                PointStruct(
                    id=_str_to_uint64(img.image_id),
                    vector=vec,
                    payload={
                        "image_id": img.image_id,
                        "page": img.page,
                        "image_path": img.image_path,
                        "image_type": img.image_type,
                        "description": img.description,
                        "keywords": img.keywords,
                        "caption": img.caption,
                    },
                )
                for img, vec in zip(batch, vectors)
            ]
            self._client.upsert(
                collection_name=settings.qdrant_image_collection,
                points=points,
            )

        console.print(f"[green]✓[/green] {len(images)} images upserted to '{settings.qdrant_image_collection}'")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search_text(
        self,
        query: str,
        top_k: int | None = None,
        page_filter: int | None = None,
    ) -> list[SearchHit]:
        """Semantic search over manual text chunks."""
        k = top_k or settings.top_k_text
        vec = self._embed([query])[0]

        query_filter = None
        if page_filter is not None:
            query_filter = Filter(
                must=[FieldCondition(key="page", match=MatchValue(value=page_filter))]
            )

        results = self._client.search(
            collection_name=settings.qdrant_text_collection,
            query_vector=vec,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [SearchHit(id=str(r.id), score=r.score, payload=r.payload) for r in results]

    def search_images(
        self,
        query: str,
        top_k: int | None = None,
        image_type: str | None = None,
    ) -> list[SearchHit]:
        """Semantic search over image caption embeddings."""
        k = top_k or settings.top_k_images
        vec = self._embed([query])[0]

        query_filter = None
        if image_type:
            query_filter = Filter(
                must=[FieldCondition(key="image_type", match=MatchValue(value=image_type))]
            )

        results = self._client.search(
            collection_name=settings.qdrant_image_collection,
            query_vector=vec,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [SearchHit(id=str(r.id), score=r.score, payload=r.payload) for r in results]

    # ── Utility ────────────────────────────────────────────────────────────────

    def collection_stats(self) -> dict:
        text_info = self._client.get_collection(settings.qdrant_text_collection)
        img_info = self._client.get_collection(settings.qdrant_image_collection)
        return {
            "text_vectors": text_info.vectors_count,
            "image_vectors": img_info.vectors_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _str_to_uint64(s: str) -> int:
    """
    Qdrant integer point IDs must fit in uint64.
    We hash the string ID deterministically.
    """
    import hashlib
    h = hashlib.sha256(s.encode()).digest()
    # Take first 8 bytes, mask to positive int64 range
    val = int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF
    return val
