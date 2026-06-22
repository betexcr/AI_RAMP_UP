# AI Ramp Up

A Flask learning project that explores the OpenAI **Responses API**: structured outputs, tool calling, embeddings, and a simple RAG pipeline over a WW2 history PDF.

## Features

- **Responses API basics** — text generation with custom instructions and prompt caching
- **Structured outputs** — Pydantic schemas for calendar events, math tutoring steps, and weather
- **Function calling** — live weather via Open-Meteo with hierarchical geocoding (Open-Meteo + Nominatim fallback)
- **Semantic search** — cosine similarity over 1,000 embedded Amazon food reviews
- **RAG** — retrieve WW2 PDF chunks by embedding similarity, then answer with GPT-4o (Responses API)

## Requirements

- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Pre-built parquet files in `data/` (included in the repo), or run the prep scripts below to regenerate them

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the environment template and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
flask --app app run --debug
```

Or use the VS Code launch config **Flask: Run main** (`.vscode/launch.json`). On Windows, `--no-reload` is enabled in the launch config to avoid socket reload issues.

The server starts at `http://127.0.0.1:5000`.

## Response format

Successful responses use a standard envelope:

```json
{ "data": { ... } }
```

Errors use:

```json
{ "error": { "code": "error_code", "message": "Human-readable message" } }
```

## API Endpoints

### `GET /` and `GET /health`

API index and health check (`reviews_loaded`, `ww2_loaded`).

### `GET /docs`

JSON index of all endpoints and parameters.

### `/poet` — basic text generation

```
GET /poet?prompt=Write+a+haiku+about+rain
GET /poet?raw=true
```

Returns `{ "data": { "text": "..." } }` by default, or the full API payload when `raw=true`.

### `/instructions` — custom system instructions

Cached `instructions/prompt.txt` as system prompt.

```
GET /instructions?question=How+do+I+declare+a+string
POST /instructions  { "question": "..." }
```

### `/calendar` and `/math` — structured output

```
GET /calendar?text=Alice+and+Bob+science+fair+Friday
POST /calendar  { "text": "..." }

GET /math?text=solve+8x+%2B+7+%3D+-23
POST /math  { "text": "..." }
```

### `/weather` — tool calling + structured output

| Parameter   | Description                                      | Default              |
|------------|--------------------------------------------------|----------------------|
| `location` | Comma-separated place hierarchy                  | —                    |
| `lat`      | Latitude (requires `lon`)                        | —                    |
| `lon`      | Longitude (requires `lat`)                       | —                    |
| `units`    | `celsius` or `fahrenheit`                        | `celsius`            |

**Resolution order:** `location` → `lat`/`lon` → default coordinates.

```
/weather?location=Santa Cruz,Turrialba,Cartago,Costa Rica
/weather?location=Cartago,Costa Rica
/weather?lat=9.9672&lon=-83.7343
```

### `/search_reviews` — embedding search

```
GET /search_reviews?search_query=spicy+noodles
POST /search_reviews  { "search_query": "..." }
```

### `/ask_ww2_history` — RAG over a PDF

```
GET /ask_ww2_history?question=When+did+D-Day+happen
POST /ask_ww2_history  { "question": "..." }
```

## Data preparation (optional)

Pre-built parquet files are committed under `data/`. To regenerate:

**Amazon reviews** (requires Kaggle credentials for `kagglehub`):

```bash
python prepare_data.py
```

**WW2 PDF chunks**:

```bash
python prepare_pdf.py
```

## Project structure

```
AI_RAMP_UP/
├── app.py                  # Entry point (create_app)
├── factory.py              # Application factory
├── config.py               # Configuration classes
├── errors.py               # API errors and global handlers
├── request_helpers.py      # Shared request/response helpers
├── schemas.py              # Pydantic request/response models
├── routes/
│   ├── demo.py             # /poet, /instructions, /calendar, /math
│   ├── weather.py          # /weather
│   ├── search.py           # /search_reviews, /ask_ww2_history
│   └── health.py           # /, /health, /docs
├── services/
│   ├── data_store.py       # Parquet loading at startup
│   ├── openai_client.py    # Shared OpenAI client
│   ├── openai_helpers.py   # Structured parse helper
│   ├── weather_service.py  # Tool loop + geocoding
│   ├── embedding_search.py # Vector similarity search
│   └── rag.py              # WW2 RAG pipeline
├── tools/tools.py
├── utils/weather.py
├── tests/
├── prepare_data.py
└── prepare_pdf.py
```

## Tests

```bash
pytest tests -v
```

Uses `FLASK_CONFIG=testing` via the `TestingConfig` (no real OpenAI calls in unit tests).

## Models used

| Use case              | Model                    |
|-----------------------|--------------------------|
| Most endpoints        | `gpt-4o-mini`            |
| WW2 RAG answers       | `gpt-4o`                 |
| Embeddings            | `text-embedding-3-small` |

## External APIs

- [OpenAI](https://platform.openai.com/) — chat, responses, embeddings
- [Open-Meteo](https://open-meteo.com/) — weather forecast and geocoding (no API key)
- [Nominatim](https://nominatim.openstreetmap.org/) — fallback geocoding for fine-grained places
