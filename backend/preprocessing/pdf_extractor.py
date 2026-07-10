"""
PDF Extraction — delegates entirely to `unstructured`.
unstructured partitions text, tables, and images in one pass,
handles chunking-aware element types, and preserves page metadata.
"""
from __future__ import annotations
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_dict

from backend.config import settings


def extract(pdf_path: Path) -> tuple[list[Document], list[dict]]:
    """
    Returns:
        docs   — LangChain Documents ready for vector store ingestion
        tables — raw dicts ready for TableStore ingestion
    """
    logger.info(f"Partitioning {pdf_path.name} …")

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                   # uses layout model for table/image detection
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(settings.images_dir),
        infer_table_structure=True,          # gives us Table elements with .metadata.text_as_html
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    docs: list[Document] = []
    tables: list[dict] = []

    for el in elements:
        meta = {"page": el.metadata.page_number, "source": pdf_path.name,
                "category": el.category}

        if el.category == "Table":
            tables.append({
                "page": el.metadata.page_number,
                "section": el.metadata.section or f"Page {el.metadata.page_number}",
                "html": el.metadata.text_as_html or "",
                "text": el.text,
            })

        elif el.category in ("NarrativeText", "Title", "ListItem", "UncategorizedText"):
            for chunk in splitter.create_documents([el.text], metadatas=[meta]):
                docs.append(chunk)

    logger.success(f"Extracted {len(docs)} chunks, {len(tables)} tables from {len(elements)} elements")
    return docs, tables
