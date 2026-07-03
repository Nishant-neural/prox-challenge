"""
Manual Processing Pipeline
==========================
Extracts four artefact types from the Vulcan OmniPro 220 PDF:

  1. Text chunks   → chunked, section-tagged, ready for Qdrant
  2. Tables        → structured JSON, ready for SQLite
  3. Images        → saved to knowledge/images/, captioned with Claude Vision
  4. Screenshots   → full-page renders saved to knowledge/screenshots/

Run via:  python scripts/ingest.py --pdf manuals/pdf/vulcan_omnipro_220.pdf
"""

from __future__ import annotations

import base64
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator

import anthropic
import fitz                 # pymupdf
import pdfplumber
from PIL import Image
from rich.console import Console
from rich.progress import track
from tenacity import retry, stop_after_attempt, wait_exponential

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextChunk:
    chunk_id: str
    page: int
    section: str
    content: str
    char_start: int
    char_end: int
    source_pdf: str


@dataclass
class TableRecord:
    table_id: str
    page: int
    section: str
    caption: str
    columns: list[str]
    rows: list[dict]                   # [{col: value, …}, …]
    raw_text: str


@dataclass
class ImageRecord:
    image_id: str
    page: int
    image_path: str                    # relative to knowledge/images/
    image_type: str                    # "diagram" | "photo" | "schematic" | "wiring" | "weld_sample" | "unknown"
    description: str
    keywords: list[str]
    caption: str


@dataclass
class ExtractionResult:
    chunks: list[TextChunk] = field(default_factory=list)
    tables: list[TableRecord] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)
    page_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _detect_section(text: str, page: int) -> str:
    """
    Heuristic section detector from page text.
    Looks for lines that look like headings (ALL CAPS, or Title Case short line).
    """
    for line in text.split("\n")[:8]:
        line = line.strip()
        if 4 < len(line) < 60:
            if line.isupper() or re.match(r"^[A-Z][a-z].*[a-z]$", line):
                return line
    return f"Page {page}"


def _chunk_text(
    text: str,
    page: int,
    section: str,
    source_pdf: str,
    chunk_size_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size_chars, text_len)
        # Try to break at a sentence boundary
        if end < text_len:
            last_period = text.rfind(".", start, end)
            if last_period > start + chunk_size_chars // 2:
                end = last_period + 1

        snippet = text[start:end].strip()
        if len(snippet) >= min_chars:
            chunk_id = f"p{page:04d}_c{idx:03d}"
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                page=page,
                section=section,
                content=snippet,
                char_start=start,
                char_end=end,
                source_pdf=source_pdf,
            ))
            idx += 1

        start = end - overlap_chars if end < text_len else text_len

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Claude Vision — image captioning
# ═══════════════════════════════════════════════════════════════════════════════

_CAPTION_PROMPT = """
You are analyzing a page image from a Vulcan OmniPro 220 multi-process welder manual.

Return ONLY a JSON object with this exact shape:
{
  "type": "<one of: diagram, photo, schematic, wiring, weld_sample, table, text, unknown>",
  "description": "<1-3 sentences describing what is shown>",
  "keywords": ["<keyword1>", "<keyword2>", ...],
  "caption": "<short caption, ≤ 15 words>"
}

Be specific: mention part names, process names (MIG, TIG, Stick, Flux Core), connector names, etc.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _caption_image(image_path: Path, client: anthropic.Anthropic) -> dict:
    """Send image to Claude Vision and parse the structured response."""
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()

    # Detect media type
    suffix = image_path.suffix.lower()
    media_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = media_map.get(suffix, "image/png")

    response = client.messages.create(
        model=settings.vision_model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                },
                {"type": "text", "text": _CAPTION_PROMPT},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Core extractor
# ═══════════════════════════════════════════════════════════════════════════════

class ManualExtractor:
    """
    Orchestrates full extraction from a single PDF file.

    Usage:
        extractor = ManualExtractor(pdf_path)
        result = extractor.extract()
    """

    def __init__(self, pdf_path: Path, caption_images: bool = True):
        self.pdf_path = pdf_path
        self.caption_images = caption_images
        self.images_dir = settings.images_dir
        self.screenshots_dir = settings.screenshots_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if caption_images else None
        self._fitz_doc = fitz.open(str(pdf_path))

    # ── Public ─────────────────────────────────────────────────────────────────

    def extract(self) -> ExtractionResult:
        result = ExtractionResult(page_count=len(self._fitz_doc))
        console.rule("[bold cyan]Phase 1 — Text & Tables")
        result.chunks, result.tables = self._extract_text_and_tables()

        console.rule("[bold cyan]Phase 2 — Page Screenshots")
        self._render_page_screenshots()

        console.rule("[bold cyan]Phase 3 — Images")
        result.images = self._extract_images()

        self._fitz_doc.close()
        return result

    # ── Text + Tables ──────────────────────────────────────────────────────────

    def _extract_text_and_tables(self) -> tuple[list[TextChunk], list[TableRecord]]:
        chunks: list[TextChunk] = []
        tables: list[TableRecord] = []
        chunk_chars = settings.chunk_size * 4
        overlap_chars = settings.chunk_overlap * 4

        with pdfplumber.open(str(self.pdf_path)) as pdf:
            for page_num, page in enumerate(
                track(pdf.pages, description="Extracting text/tables…"), start=1
            ):
                # ── Extract tables first (pdfplumber is great at this) ──────────
                for t_idx, table in enumerate(page.extract_tables()):
                    if not table or len(table) < 2:
                        continue
                    header = [str(cell or "").strip() for cell in table[0]]
                    rows = []
                    for row in table[1:]:
                        row_dict = {
                            col: str(cell or "").strip()
                            for col, cell in zip(header, row)
                        }
                        rows.append(row_dict)

                    raw_text = "\n".join(
                        " | ".join(str(c or "") for c in row) for row in table
                    )
                    section = _detect_section(page.extract_text() or "", page_num)
                    table_id = f"tbl_p{page_num:04d}_{t_idx:02d}"
                    tables.append(TableRecord(
                        table_id=table_id,
                        page=page_num,
                        section=section,
                        caption=f"Table on page {page_num}",
                        columns=header,
                        rows=rows,
                        raw_text=raw_text,
                    ))

                # ── Extract page text (minus table bounding boxes) ────────────
                # Crop out table areas so they don't double-appear in text chunks
                cropped = page
                for table_bbox in page.find_tables():
                    cropped = cropped.outside_bbox(table_bbox.bbox)

                text = cropped.extract_text(x_tolerance=3, y_tolerance=3) or ""
                if len(text.strip()) < settings.min_chunk_chars:
                    continue

                section = _detect_section(text, page_num)
                page_chunks = _chunk_text(
                    text=text,
                    page=page_num,
                    section=section,
                    source_pdf=self.pdf_path.name,
                    chunk_size_chars=chunk_chars,
                    overlap_chars=overlap_chars,
                    min_chars=settings.min_chunk_chars,
                )
                chunks.extend(page_chunks)

        console.print(f"[green]✓[/green] {len(chunks)} text chunks, {len(tables)} tables extracted")
        return chunks, tables

    # ── Page screenshots ───────────────────────────────────────────────────────

    def _render_page_screenshots(self):
        """Render every page as a PNG at 150 DPI for the Manual Viewer."""
        for page_num in track(
            range(len(self._fitz_doc)), description="Rendering page screenshots…"
        ):
            page = self._fitz_doc[page_num]
            mat = fitz.Matrix(150 / 72, 150 / 72)   # 150 DPI
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_path = self.screenshots_dir / f"page_{page_num + 1:04d}.png"
            if not out_path.exists():
                pix.save(str(out_path))

        console.print(f"[green]✓[/green] {len(self._fitz_doc)} page screenshots saved")

    # ── Images ────────────────────────────────────────────────────────────────

    def _extract_images(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        MIN_PIXELS = 80 * 80    # skip tiny icons / bullets

        for page_num in track(
            range(len(self._fitz_doc)), description="Extracting images…"
        ):
            page = self._fitz_doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = self._fitz_doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]

                # Filter out tiny images
                from io import BytesIO
                pil_img = Image.open(BytesIO(img_bytes))
                w, h = pil_img.size
                if w * h < MIN_PIXELS:
                    continue

                image_id = f"img_p{page_num + 1:04d}_{img_idx:02d}"
                filename = f"{image_id}.{ext}"
                img_path = self.images_dir / filename

                if not img_path.exists():
                    img_path.write_bytes(img_bytes)

                # Caption with Claude Vision (or stub if disabled)
                if self.caption_images and self._client:
                    try:
                        meta = _caption_image(img_path, self._client)
                    except Exception as exc:
                        console.print(f"[yellow]⚠ captioning failed for {filename}: {exc}[/yellow]")
                        meta = {
                            "type": "unknown",
                            "description": f"Image on page {page_num + 1}",
                            "keywords": [],
                            "caption": f"Image page {page_num + 1}",
                        }
                else:
                    meta = {
                        "type": "unknown",
                        "description": f"Image on page {page_num + 1}",
                        "keywords": [],
                        "caption": f"Image page {page_num + 1}",
                    }

                records.append(ImageRecord(
                    image_id=image_id,
                    page=page_num + 1,
                    image_path=str(img_path.relative_to(settings.knowledge_dir)),
                    image_type=meta.get("type", "unknown"),
                    description=meta.get("description", ""),
                    keywords=meta.get("keywords", []),
                    caption=meta.get("caption", ""),
                ))

        console.print(f"[green]✓[/green] {len(records)} images extracted and captioned")
        return records

    # ── Serialisation ─────────────────────────────────────────────────────────

    @staticmethod
    def save_result(result: ExtractionResult, knowledge_dir: Path):
        """Persist chunks and images to JSON for inspection / re-ingestion."""
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = knowledge_dir / "chunks.json"
        chunks_path.write_text(
            json.dumps([asdict(c) for c in result.chunks], indent=2)
        )

        images_path = knowledge_dir / "images_metadata.json"
        images_path.write_text(
            json.dumps([asdict(i) for i in result.images], indent=2)
        )

        # tables are persisted by the TableStore directly
        console.print(f"[green]✓[/green] Saved chunks.json and images_metadata.json to {knowledge_dir}")
