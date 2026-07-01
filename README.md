# Vulcan OmniPro 220 — Multimodal Expert Agent

An AI welding assistant that behaves like an experienced technician standing beside you.
Built on Claude + Qdrant + SQLite with a React frontend for interactive artifacts.

---

## Architecture

```
User
 │
 ▼
React Frontend  (Vite + Tailwind)
 │  Chat UI + Artifact renderer
 │
 ▼ SSE stream
FastAPI Backend
 │
 ├─ Claude Agent (tool_use loop)
 │     ├── search_manual      → Qdrant text collection
 │     ├── lookup_table       → SQLite exact query
 │     ├── find_diagram       → Qdrant image collection
 │     ├── get_manual_page    → Screenshot files
 │     └── generate_artifact  → Frontend artifact signal
 │
 ├─ Qdrant  (manual_text + manual_images)
 ├─ SQLite  (tables.db — duty cycles, settings, specs)
 └─ knowledge/  (images/, screenshots/, chunks.json)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node 18+
- Docker (for Qdrant)

### 2. Clone & install

```bash
git clone <repo>
cd omnipro-agent

# Backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
```

### 4. Start Qdrant

```bash
docker compose up -d
```

### 5. Ingest the manual

Place the PDF at `manuals/pdf/vulcan_omnipro_220.pdf`, then:

```bash
python scripts/ingest.py --pdf manuals/pdf/vulcan_omnipro_220.pdf
```

This will:
- Extract all text chunks, tables, images, and page screenshots
- Caption images with Claude Vision
- Upsert embeddings into Qdrant
- Store structured tables in SQLite

Takes ~5–15 minutes depending on PDF size and image count.

**Flags:**
```
--no-captions   Skip Claude Vision (faster; images unsearchable)
--reset         Drop and recreate Qdrant collections
--skip-images   Skip image extraction entirely
```

### 6. Run the backend

```bash
uvicorn backend.app:app --reload
# → http://localhost:8000
# → http://localhost:8000/api/health
```

### 7. Run the frontend

```bash
cd frontend
npm run dev
# → http://localhost:5173
```

---

## Project Structure

```
omnipro-agent/
├── backend/
│   ├── config.py              # All settings (Pydantic + .env)
│   ├── app.py                 # FastAPI — SSE /api/chat + image serving
│   ├── agent.py               # Claude agentic loop (tool_use)
│   ├── tools/
│   │   └── manual_tools.py    # Tool schemas + ToolExecutor
│   ├── retrieval/
│   │   ├── vector_store.py    # Qdrant text + image search
│   │   └── table_store.py     # SQLite table lookup
│   └── preprocessing/
│       └── pdf_extractor.py   # PDF → chunks + tables + images + screenshots
├── frontend/
│   └── src/
│       ├── components/        # Chat, Message, SourceCitation, ImagePreview
│       ├── artifacts/         # DutyCycleCalculator, PolarityDiagram, etc.
│       └── pages/             # Main chat page
├── scripts/
│   └── ingest.py              # CLI ingestion runner
├── knowledge/                 # Generated — chunks.json, images/, screenshots/
├── manuals/pdf/               # Drop your PDF here
├── docker-compose.yml         # Qdrant
├── .env.example
└── requirements.txt
```

---

## SSE Event Types

The `/api/chat` endpoint streams Server-Sent Events:

| Event type | Description |
|---|---|
| `text_delta` | Incremental Claude text chunk |
| `tool_call` | Claude decided to call a tool |
| `tool_result` | Tool execution result (log) |
| `artifact` | Render an interactive widget |
| `done` | Final event with token usage + artifact list |
| `error` | Something went wrong |

---

## Artifacts

| Artifact type | Trigger |
|---|---|
| `duty_cycle_calculator` | "What's the duty cycle at 150A?" |
| `duty_cycle_visualizer` | "Show me the duty cycle" |
| `polarity_diagram` | "How do I set polarity for TIG?" |
| `settings_configurator` | "I'm welding 3mm steel with MIG" |
| `troubleshooting_wizard` | "My weld has porosity" |
| `wire_feed_explainer` | "How does the wire feed work?" |

---

## Day-by-Day Build Plan

| Day | Focus |
|-----|-------|
| 1 | Repo + pipeline + Qdrant + SQLite + ingest script ✅ |
| 2 | Image extraction + Vision captions + agent tools + table lookup |
| 3 | Artifacts: DutyCycleCalc, PolarityDiagram, TroubleshootingWizard, SettingsConfigurator |
| 4 | Frontend polish + streaming + Manual Viewer + README + demo video |
