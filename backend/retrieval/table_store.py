"""
SQLite Table Store
==================
Stores all structured tables extracted from the manual.
Supports exact-match queries by process, material, thickness, voltage, current.

Schema
------
  tables      — one row per extracted table, JSON payload
  table_rows  — one row per data row, with indexed columns for fast lookup

Usage
-----
  store = TableStore()
  store.ingest_tables(tables)
  results = store.query(process="MIG", material="steel", thickness_mm=3.0)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.preprocessing.pdf_extractor import TableRecord

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════════

_DDL = """
CREATE TABLE IF NOT EXISTS tables (
    table_id    TEXT PRIMARY KEY,
    page        INTEGER,
    section     TEXT,
    caption     TEXT,
    columns     TEXT,   -- JSON list
    rows        TEXT,   -- JSON list of dicts
    raw_text    TEXT
);

-- Denormalized flat rows for lookup queries
CREATE TABLE IF NOT EXISTS table_rows (
    row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id    TEXT REFERENCES tables(table_id),
    page        INTEGER,
    section     TEXT,

    -- Common welding parameters (NULL if not present in this table)
    process     TEXT,   -- MIG, TIG, Stick, Flux Core
    material    TEXT,   -- steel, stainless, aluminum, …
    thickness   REAL,   -- mm
    voltage     REAL,
    current     REAL,
    wire_size   TEXT,
    gas_mix     TEXT,
    duty_cycle  REAL,

    raw_row     TEXT    -- full JSON of the original row dict
);

CREATE INDEX IF NOT EXISTS idx_rows_process   ON table_rows(process);
CREATE INDEX IF NOT EXISTS idx_rows_material  ON table_rows(material);
CREATE INDEX IF NOT EXISTS idx_rows_thickness ON table_rows(thickness);
"""

# Keyword → canonical column name mapping
_FIELD_ALIASES: dict[str, str] = {
    # process
    "process": "process", "welding process": "process", "type": "process",
    # material
    "material": "material", "base metal": "material", "metal": "material",
    # thickness
    "thickness": "thickness", "thick": "thickness", "gauge": "thickness",
    "thickness (mm)": "thickness", "thickness mm": "thickness",
    # voltage
    "voltage": "voltage", "volts": "voltage", "v": "voltage",
    # current
    "current": "current", "amps": "current", "amperage": "current", "a": "current",
    # wire
    "wire size": "wire_size", "wire diameter": "wire_size", "electrode": "wire_size",
    # gas
    "gas": "gas_mix", "shielding gas": "gas_mix", "gas mix": "gas_mix",
    # duty cycle
    "duty cycle": "duty_cycle", "duty": "duty_cycle",
}

_PROCESS_KEYWORDS: dict[str, list[str]] = {
    "MIG": ["mig", "gmaw", "wire feed", "wire-feed"],
    "TIG": ["tig", "gtaw", "tungsten"],
    "Stick": ["stick", "smaw", "electrode", "shielded metal"],
    "Flux Core": ["flux", "fcaw", "flux-cored", "flux core"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Store
# ═══════════════════════════════════════════════════════════════════════════════

class TableStore:
    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or settings.sqlite_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(_DDL)
        self._con.commit()

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_tables(self, tables: list[TableRecord]):
        cur = self._con.cursor()
        inserted_tables = 0
        inserted_rows = 0

        for tbl in tables:
            # Skip if already ingested
            existing = cur.execute(
                "SELECT 1 FROM tables WHERE table_id = ?", (tbl.table_id,)
            ).fetchone()
            if existing:
                continue

            cur.execute(
                """
                INSERT INTO tables (table_id, page, section, caption, columns, rows, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tbl.table_id,
                    tbl.page,
                    tbl.section,
                    tbl.caption,
                    json.dumps(tbl.columns),
                    json.dumps(tbl.rows),
                    tbl.raw_text,
                ),
            )
            inserted_tables += 1

            # Flatten rows into table_rows
            for row_dict in tbl.rows:
                flat = _flatten_row(row_dict, tbl)
                cur.execute(
                    """
                    INSERT INTO table_rows
                        (table_id, page, section, process, material, thickness,
                         voltage, current, wire_size, gas_mix, duty_cycle, raw_row)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tbl.table_id,
                        tbl.page,
                        tbl.section,
                        flat.get("process"),
                        flat.get("material"),
                        flat.get("thickness"),
                        flat.get("voltage"),
                        flat.get("current"),
                        flat.get("wire_size"),
                        flat.get("gas_mix"),
                        flat.get("duty_cycle"),
                        json.dumps(row_dict),
                    ),
                )
                inserted_rows += 1

        self._con.commit()
        console.print(
            f"[green]✓[/green] Ingested {inserted_tables} tables / {inserted_rows} rows into SQLite"
        )

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        process: str | None = None,
        material: str | None = None,
        thickness_mm: float | None = None,
        voltage: float | None = None,
        current: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Exact / fuzzy lookup of welding parameters.
        All parameters are optional; any combination is valid.
        Returns raw_row dicts plus page and section for citation.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if process:
            canonical = _normalize_process(process)
            if canonical:
                clauses.append("process = ?")
                params.append(canonical)

        if material:
            clauses.append("LOWER(material) LIKE ?")
            params.append(f"%{material.lower()}%")

        if thickness_mm is not None:
            # Allow ±15% tolerance on thickness
            lo, hi = thickness_mm * 0.85, thickness_mm * 1.15
            clauses.append("thickness BETWEEN ? AND ?")
            params.extend([lo, hi])

        if voltage is not None:
            lo, hi = voltage - 2, voltage + 2
            clauses.append("voltage BETWEEN ? AND ?")
            params.extend([lo, hi])

        if current is not None:
            lo, hi = current * 0.85, current * 1.15
            clauses.append("current BETWEEN ? AND ?")
            params.extend([lo, hi])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT tr.page, tr.section, tr.raw_row,
                   t.caption, t.table_id
            FROM table_rows tr
            JOIN tables t USING (table_id)
            {where}
            LIMIT ?
        """
        params.append(limit)
        rows = self._con.execute(sql, params).fetchall()

        results = []
        for row in rows:
            data = json.loads(row["raw_row"])
            results.append({
                "table_id": row["table_id"],
                "page": row["page"],
                "section": row["section"],
                "caption": row["caption"],
                "data": data,
            })
        return results

    def get_table_by_id(self, table_id: str) -> dict | None:
        row = self._con.execute(
            "SELECT * FROM tables WHERE table_id = ?", (table_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "table_id": row["table_id"],
            "page": row["page"],
            "section": row["section"],
            "caption": row["caption"],
            "columns": json.loads(row["columns"]),
            "rows": json.loads(row["rows"]),
        }

    def get_all_tables_summary(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT table_id, page, section, caption FROM tables ORDER BY page"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self._con.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_process(raw: str) -> str | None:
    low = raw.lower().strip()
    for canonical, keywords in _PROCESS_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return canonical
    return None


def _parse_numeric(value: str) -> float | None:
    """Extract first numeric value from a cell like '18-22V' or '200A'."""
    import re
    m = re.search(r"[\d]+\.?[\d]*", str(value))
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def _flatten_row(row: dict, tbl: TableRecord) -> dict[str, Any]:
    """
    Map raw column headers to canonical field names and extract numeric values.
    Also infer process from section/caption if not a column.
    """
    flat: dict[str, Any] = {}

    for col, val in row.items():
        canonical = _FIELD_ALIASES.get(col.lower().strip())
        if not canonical:
            continue
        if canonical in ("thickness", "voltage", "current", "duty_cycle"):
            flat[canonical] = _parse_numeric(str(val))
        else:
            flat[canonical] = str(val).strip() if val else None

    # Infer process from section or caption if not a column
    if "process" not in flat:
        combined = f"{tbl.section} {tbl.caption}".lower()
        proc = _normalize_process(combined)
        if proc:
            flat["process"] = proc

    return flat
