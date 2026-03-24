# Credit Guard

Credit Guard estimates credit-card default risk and surfaces why the model said what it said: XGBoost (or another trained tree model) for the score, SHAP for feature impact, and an optional LLM (Groq or Ollama) to turn numbers into short text.

![CI](https://github.com/YuukiFST/Credit-Guard/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## What you get

- **REST API** (FastAPI): `/predict`, `/explain`, `/health`. Open `/docs` in the browser to try requests.
- **Streamlit UI** (`app/streamlit_app.py`): forms and charts; it calls the API over HTTP.
- **Training script** (`python -m src.models.trainer`): needs `data/raw/UCI_Credit_Card.csv` and writes `models/best_model.pkl` (gitignored; train locally or supply your own artifact).

## Requirements

- Python 3.11+
- Dependencies: `requirements.txt` (app) and `requirements-dev.txt` (tests, lint, types)
- **Groq API key** in `.env` if you want `/explain` to use the cloud LLM path (see `.env.example`). Ollama remains an alternative if you run it yourself.

## Quick start (no Docker)

```bash
git clone https://github.com/YuukiFST/Credit-Guard.git
cd Credit-Guard
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# Edit .env: set GROQ_API_KEY if you use Groq
```

Train once if you do not have `models/best_model.pkl`:

```bash
python -m src.models.trainer
```

Run the API (pick a free port if 8000 is blocked on your machine):

```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
```

Run Streamlit in a second terminal. If the API is not on port 8000, set the base URL:

```bash
export CREDITGUARD_API_URL=http://127.0.0.1:8001   # macOS/Linux
# Windows PowerShell: $env:CREDITGUARD_API_URL = "http://127.0.0.1:8001"
python -m streamlit run app/streamlit_app.py
```

With **Make** (if `make` and `uv` are available):

```bash
make dev
make train    # optional
make api
make app      # in another shell
```

## Docker (optional)

```bash
cp .env.example .env
docker compose up -d
```

Use this if you prefer containers over a local venv.

## How the pieces connect

1. Load UCI-style rows, engineer features, train or load a serialized model.
2. API validates JSON with Pydantic, runs the model and SHAP, optionally calls the LLM client.
3. Streamlit is a thin client: it POSTs the same JSON shape to `/predict` and `/explain`.

Audit decisions go to JSONL under `audit_trail/` (ignored by git by default).

## Repo layout

- `src/api` — FastAPI app and routes
- `src/audit` — JSONL audit logger
- `src/config.py` — settings (YAML + env)
- `src/data` — Pydantic schemas and loaders
- `src/explainability` — SHAP helper
- `src/features` — feature engineering
- `src/llm` — LLM client and guardrails
- `src/models` — training, metrics, calibration helpers
- `app` — Streamlit
- `tests` — pytest

For a longer architecture write-up, keep a private copy outside the repo (for example under a local-only `resources/` folder that you do not commit) or extend this README in your fork.

## Tests

```bash
pytest
```

`pyproject.toml` sets coverage options (including the minimum gate). If `make test` fails, use `pytest` directly or align the Makefile with the same `--cov-fail-under` as in `pyproject.toml`.

## License

MIT. See [LICENSE](LICENSE).
