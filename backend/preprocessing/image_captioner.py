"""
Generate searchable captions for extracted manual images.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path

import anthropic
from langchain_core.documents import Document
from loguru import logger

from backend.config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

_PAGE_PATTERN = re.compile(r"page[-_]?(\d+)", re.IGNORECASE)


def caption_images(image_dir: Path, discovered_entries: list[dict] | None = None) -> tuple[list[Document], list[dict]]:
    image_docs: list[Document] = []
    metadata: list[dict] = []
    discovered_lookup = _build_lookup(discovered_entries or [])

    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue

        page = _page_from_name(path.name)
        caption = _caption_with_fallback(path)
        discovered = discovered_lookup.get(path.name, {})
        entry = {
            "image_path": str(path.relative_to(settings.knowledge_dir)),
            "file_name": path.name,
            "page": discovered.get("page", page),
            "section": discovered.get("section"),
            "type": caption["type"],
            "description": caption["description"],
            "keywords": caption["keywords"],
        }
        metadata.append(entry)
        image_docs.append(
            Document(
                page_content=_caption_text(entry),
                metadata=entry,
            )
        )

    logger.success(f"Prepared {len(image_docs)} image captions")
    return image_docs, metadata


def _caption_with_fallback(path: Path) -> dict:
    if _client is None:
        return _fallback_caption(path)

    try:
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        response = _client.messages.create(
            model=settings.vision_model,
            max_tokens=250,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this welding manual image as compact JSON with keys "
                                "type, description, and keywords. Keep keywords to 3-8 short terms."
                            ),
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": _b64(path),
                            },
                        },
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        parsed = json.loads(_extract_json(text))
        if not isinstance(parsed.get("keywords"), list):
            parsed["keywords"] = []
        return {
            "type": str(parsed.get("type") or "manual_image"),
            "description": str(parsed.get("description") or path.stem.replace("_", " ")),
            "keywords": [str(item) for item in parsed["keywords"][:8]],
        }
    except Exception as exc:
        logger.warning(f"Vision captioning failed for {path.name}: {exc}")
        return _fallback_caption(path)


def _fallback_caption(path: Path) -> dict:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    words = [token for token in stem.split() if token]
    return {
        "type": _guess_type(stem),
        "description": stem or "manual image",
        "keywords": words[:8],
    }


def _guess_type(text: str) -> str:
    lowered = text.lower()
    if "wiring" in lowered or "cable" in lowered:
        return "wiring"
    if "diagram" in lowered or "schematic" in lowered:
        return "diagram"
    if "photo" in lowered or "panel" in lowered:
        return "photo"
    return "manual_image"


def _caption_text(entry: dict) -> str:
    keyword_text = ", ".join(entry["keywords"])
    return f"{entry['type']}. {entry['description']}. Keywords: {keyword_text}"


def _page_from_name(name: str) -> int | None:
    match = _PAGE_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _build_lookup(entries: list[dict]) -> dict[str, dict]:
    lookup = {}
    for entry in entries:
        raw_path = entry.get("image_path")
        if not raw_path:
            continue
        lookup[Path(raw_path).name] = entry
    return lookup


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in caption response")
    return text[start : end + 1]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
