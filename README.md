# Credit Guard

Credit Guard predicts credit card default risk and explains its decisions. It uses XGBoost for the prediction, SHAP to extract the mathematical reasons, and an LLM to translate those reasons into text.

![CI](https://github.com/seu-usuario/CreditGuard/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Quick Start

You need a Groq API key to run the LLM fallback. Get one at console.groq.com and add it to your `.env` file.

```bash
git clone https://github.com/seu-usuario/CreditGuard.git
cd CreditGuard
cp .env.example .env
docker-compose up -d
```

To run it locally without Docker:

```bash
make dev
make train
make api
make app
```

## Architecture

1. **Pipeline**: Loads the UCI Credit Card dataset, runs SMOTE for class balance, and trains the models.
2. **Calibration**: Adjusts probabilities so a 0.8 score means an 80% chance of default.
3. **API**: A FastAPI service takes client data, runs the calibrated model, and generates SHAP values.
4. **Explanation**: An LLM (Ollama or Groq) reads the SHAP values and writes a summary.

An audit trail saves every decision as JSONL.

## Project Structure

- `src/api`: FastAPI and routing
- `src/audit`: JSONL decision logs
- `src/config.py`: Centralized settings
- `src/data`: Pydantic schemas and data loaders
- `src/explainability`: SHAP integration
- `src/features`: Feature engineering
- `src/llm`: Groq/Ollama clients and guardrails
- `src/models`: Training, fairness, and drift monitoring
- `app`: Streamlit frontend
- `tests`: Pytest suite

## Tests

The CI pipeline runs pytest with coverage tracking. To run them locally:

```bash
make test
```

## License

MIT License. See [LICENSE](LICENSE).
