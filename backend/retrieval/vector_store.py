"""
Vector Store — thin wrapper around langchain-qdrant.
Two named collections: text chunks and image captions.
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


def _ensure_collection(client: QdrantClient, name: str):
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(name, vectors_config=VectorParams(
            size=settings.embed_dim, distance=Distance.COSINE,
        ))
        logger.info(f"Created collection: {name}")


def get_text_store() -> QdrantVectorStore:
    client = _client()
    _ensure_collection(client, settings.qdrant_text_collection)
    return QdrantVectorStore(client=client, collection_name=settings.qdrant_text_collection,
                             embedding=_embeddings)


def get_image_store() -> QdrantVectorStore:
    client = _client()
    _ensure_collection(client, settings.qdrant_image_collection)
    return QdrantVectorStore(client=client, collection_name=settings.qdrant_image_collection,
                             embedding=_embeddings)


def ingest_docs(docs: list[Document]):
    store = get_text_store()
    store.add_documents(docs)
    logger.success(f"Upserted {len(docs)} text chunks")


def ingest_image_captions(captions: list[Document]):
    store = get_image_store()
    store.add_documents(captions)
    logger.success(f"Upserted {len(captions)} image captions")


def search_text(query: str, k: int = None) -> list[Document]:
    return get_text_store().similarity_search(query, k=k or settings.top_k_text)


def search_images(query: str, k: int = None) -> list[Document]:
    return get_image_store().similarity_search(query, k=k or settings.top_k_images)
