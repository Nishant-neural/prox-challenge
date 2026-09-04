"""
Vector store helpers for text and image retrieval.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from backend.config import settings

_embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)


def _client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _ensure_collection(client: QdrantClient, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            name,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {name}")


def _get_store(collection_name: str) -> QdrantVectorStore:
    client = _client()
    _ensure_collection(client, collection_name)
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=_embeddings)


def get_text_store() -> QdrantVectorStore:
    return _get_store(settings.qdrant_text_collection)


def get_image_store() -> QdrantVectorStore:
    return _get_store(settings.qdrant_image_collection)


def ingest_docs(docs: list[Document]) -> None:
    if not docs:
        return
    get_text_store().add_documents(docs)
    logger.success(f"Upserted {len(docs)} text chunks")


def ingest_image_captions(captions: list[Document]) -> None:
    if not captions:
        return
    get_image_store().add_documents(captions)
    logger.success(f"Upserted {len(captions)} image captions")


def search_text(query: str, k: int | None = None) -> list[Document]:
    return get_text_store().similarity_search(query, k=k or settings.top_k_text)


def search_images(query: str, k: int | None = None) -> list[Document]:
    return get_image_store().similarity_search(query, k=k or settings.top_k_images)


def collection_stats() -> dict:
    client = _client()
    collections = client.get_collections().collections
    summary = {}
    for collection in collections:
        info = client.get_collection(collection.name)
        summary[collection.name] = {
            "points": info.points_count,
            "vectors": info.vectors_count,
            "status": str(info.status),
        }
    return summary


class VectorStore:
    def collection_stats(self) -> dict:
        return collection_stats()
