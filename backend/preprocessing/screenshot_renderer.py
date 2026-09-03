"""
Render each PDF page to a PNG screenshot for manual citations.
"""
from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger

from backend.config import settings


def render_page_screenshots(pdf_path: Path) -> list[dict]:
    settings.screenshots_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    rendered_pages: list[dict] = []

    for index in range(len(pdf)):
        page_number = index + 1
        bitmap = pdf[index].render(scale=settings.screenshot_scale)
        image = bitmap.to_pil()
        output_path = settings.screenshots_dir / f"page_{page_number:04d}.png"
        image.save(output_path)
        rendered_pages.append(
            {
                "page": page_number,
                "path": str(output_path.relative_to(settings.knowledge_dir)),
                "width": image.width,
                "height": image.height,
            }
        )

    logger.success(f"Rendered {len(rendered_pages)} page screenshots")
    return rendered_pages
