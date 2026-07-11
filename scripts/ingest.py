#!/usr/bin/env python3
"""
Ingest the Vulcan OmniPro 220 PDF into all knowledge stores.

    python scripts/ingest.py --pdf manuals/pdf/vulcan_omnipro_220.pdf
"""
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger
from backend.preprocessing.pdf_extractor import extract
from backend.retrieval.vector_store import ingest_docs, ingest_image_captions
from backend.retrieval.table_store import ingest_tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True, type=Path)
    args = p.parse_args()

    t0 = time.perf_counter()

    docs, tables = extract(args.pdf)
    ingest_docs(docs)
    ingest_tables(tables)

    logger.success(f"Done in {time.perf_counter() - t0:.1f}s — "
                   f"{len(docs)} chunks, {len(tables)} tables")


if __name__ == "__main__":
    main()
