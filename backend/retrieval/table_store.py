"""
Table Store — SQLAlchemy ORM over SQLite.
Structured tables from the manual (duty cycles, settings, specs).
"""
from __future__ import annotations
from typing import Any

from loguru import logger
from sqlalchemy import JSON, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from backend.config import settings


# ── ORM models ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class ManualTable(Base):
    __tablename__ = "manual_tables"

    id:      Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    page:    Mapped[int]  = mapped_column(Integer, index=True)
    section: Mapped[str]  = mapped_column(String)
    html:    Mapped[str]  = mapped_column(Text)     # raw HTML from unstructured
    text:    Mapped[str]  = mapped_column(Text)     # plain text fallback

    # Parsed welding parameters (nullable — not every table has all fields)
    process:    Mapped[str | None] = mapped_column(String, index=True)
    material:   Mapped[str | None] = mapped_column(String)
    thickness:  Mapped[float | None] = mapped_column(Float)
    voltage:    Mapped[float | None] = mapped_column(Float)
    current:    Mapped[float | None] = mapped_column(Float)
    duty_cycle: Mapped[float | None] = mapped_column(Float)
    rows:       Mapped[list | None]  = mapped_column(JSON)  # parsed rows list


# ── Engine (module-level singleton) ────────────────────────────────────────────

_engine = create_engine(f"sqlite:///{settings.sqlite_path}", echo=False)
Base.metadata.create_all(_engine)


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_tables(raw_tables: list[dict]):
    """raw_tables come directly from pdf_extractor.extract()"""
    with Session(_engine) as session:
        for t in raw_tables:
            session.add(ManualTable(
                page=t["page"],
                section=t.get("section", ""),
                html=t.get("html", ""),
                text=t.get("text", ""),
                process=_infer_process(t.get("text", "")),
                rows=_parse_html_rows(t.get("html", "")),
            ))
        session.commit()
    logger.success(f"Ingested {len(raw_tables)} tables into SQLite")


def query(
    process: str | None = None,
    material: str | None = None,
    thickness_mm: float | None = None,
    voltage: float | None = None,
    current: float | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    stmt = select(ManualTable)
    if process:
        stmt = stmt.where(ManualTable.process == _normalize_process(process))
    if thickness_mm:
        lo, hi = thickness_mm * 0.85, thickness_mm * 1.15
        stmt = stmt.where(ManualTable.thickness.between(lo, hi))
    if voltage:
        stmt = stmt.where(ManualTable.voltage.between(voltage - 2, voltage + 2))
    if current:
        stmt = stmt.where(ManualTable.current.between(current * 0.85, current * 1.15))

    with Session(_engine) as session:
        rows = session.scalars(stmt.limit(limit)).all()
        return [{"page": r.page, "section": r.section, "text": r.text,
                 "process": r.process, "rows": r.rows} for r in rows]


# ── Helpers ────────────────────────────────────────────────────────────────────

_PROCESS_MAP = {
    "MIG": ["mig", "gmaw", "wire feed"],
    "TIG": ["tig", "gtaw", "tungsten"],
    "Stick": ["stick", "smaw"],
    "Flux Core": ["flux", "fcaw", "flux-cored"],
}

def _infer_process(text: str) -> str | None:
    low = text.lower()
    return next((p for p, kws in _PROCESS_MAP.items() if any(k in low for k in kws)), None)

def _normalize_process(raw: str) -> str | None:
    return _infer_process(raw) or raw

def _parse_html_rows(html: str) -> list[dict] | None:
    if not html:
        return None
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return None
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        return [
            dict(zip(headers, [td.get_text(strip=True) for td in row.find_all("td")]))
            for row in rows[1:]
        ]
    except Exception:
        return None
