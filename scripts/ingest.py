#!/usr/bin/env python3
"""
Manual Ingestion Script
=======================
Run this once (or whenever the manual is updated) to:
  1. Extract text chunks, tables, and images from the PDF
  2. Caption all images with Claude Vision
  3. Upsert text + image embeddings into Qdrant
  4. Store structured tables in SQLite
  5. Save chunk/image metadata JSON to knowledge/

Usage:
    python scripts/ingest.py --pdf manuals/pdf/vulcan_omnipro_220.pdf
    python scripts/ingest.py --pdf manuals/pdf/vulcan_omnipro_220.pdf --no-captions
    python scripts/ingest.py --pdf manuals/pdf/vulcan_omnipro_220.pdf --reset
"""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.preprocessing.pdf_extractor import ManualExtractor
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.table_store import TableStore

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Ingest OmniPro 220 manual into knowledge stores.")
    p.add_argument("--pdf", required=True, type=Path, help="Path to the PDF manual.")
    p.add_argument(
        "--no-captions",
        action="store_true",
        help="Skip Claude Vision captioning (faster, images unsearchable).",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate Qdrant collections before ingesting.",
    )
    p.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image extraction entirely.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pdf_path = args.pdf.resolve() if not args.pdf.is_absolute() else args.pdf

    if not pdf_path.exists():
        console.print(f"[red]✗ PDF not found: {pdf_path}[/red]")
        sys.exit(1)

    if not settings.anthropic_api_key and not args.no_captions:
        console.print(
            "[yellow]⚠ ANTHROPIC_API_KEY not set. Running with --no-captions.[/yellow]"
        )
        args.no_captions = True

    console.print(Panel.fit(
        f"[bold cyan]Vulcan OmniPro 220 — Manual Ingestion[/bold cyan]\n"
        f"PDF: {pdf_path}\n"
        f"Captions: {'disabled' if args.no_captions else 'enabled (Claude Vision)'}\n"
        f"Reset stores: {args.reset}",
        border_style="cyan",
    ))

    t0 = time.time()

    # ── Step 1: Extract ────────────────────────────────────────────────────────
    console.rule("[bold]Step 1 — PDF Extraction")
    extractor = ManualExtractor(
        pdf_path=pdf_path,
        caption_images=not args.no_captions,
    )
    result = extractor.extract()
    ManualExtractor.save_result(result, settings.knowledge_dir)

    # ── Step 2: Vector store ───────────────────────────────────────────────────
    console.rule("[bold]Step 2 — Qdrant Vector Store")
    vs = VectorStore()

    if args.reset:
        from qdrant_client.http.models import Distance, VectorParams
        console.print("[yellow]Resetting Qdrant collections…[/yellow]")
        vs._client.delete_collection(settings.qdrant_text_collection)
        vs._client.delete_collection(settings.qdrant_image_collection)
        vs._ensure_collections()

    vs.upsert_chunks(result.chunks)
    if result.images and not args.skip_images:
        vs.upsert_images(result.images)

    # ── Step 3: Table store ────────────────────────────────────────────────────
    console.rule("[bold]Step 3 — SQLite Table Store")
    ts = TableStore()
    ts.ingest_tables(result.tables)
    ts.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    stats = vs.collection_stats()

    console.print(Panel.fit(
        f"[bold green]✓ Ingestion complete in {elapsed:.1f}s[/bold green]\n\n"
        f"Text chunks:    {len(result.chunks)}\n"
        f"Tables:         {len(result.tables)}\n"
        f"Images:         {len(result.images)}\n"
        f"Pages:          {result.page_count}\n\n"
        f"Qdrant text:    {stats['text_vectors']} vectors\n"
        f"Qdrant images:  {stats['image_vectors']} vectors\n",
        border_style="green",
    ))
    console.print("\nNext step: [bold]uvicorn backend.app:app --reload[/bold]")


if __name__ == "__main__":
    main()
