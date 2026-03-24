# Credit Guard

O Credit Guard estima o risco de inadimplência em cartão de crédito e mostra **porquê** o modelo decidiu o que decidiu: um modelo em árvore (tipicamente XGBoost) para a pontuação, **SHAP** para o impacto de cada variável, e um **LLM** opcional (Groq ou Ollama) para transformar números em texto curto.

![CI](https://github.com/YuukiFST/Credit-Guard/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## O que inclui

- **API REST** (FastAPI): rotas `/predict`, `/explain`, `/health`. Abre `/docs` no navegador para experimentar pedidos.
- **Interface Streamlit** (`app/streamlit_app.py`): formulários e gráficos; fala com a API por HTTP.
- **Treino** (`python -m src.models.trainer`): precisa de `data/raw/UCI_Credit_Card.csv` e grava `models/best_model.pkl` (ignorado pelo Git; treina localmente ou usa o teu próprio artefacto).

## Requisitos

- Python 3.11+
- Dependências: `requirements.txt` (aplicação) e `requirements-dev.txt` (testes, lint, tipos)
- **Chave Groq** no `.env` se quiseres que `/explain` use o LLM na nuvem (vê `.env.example`). **Ollama** continua como alternativa se o tiveres a correr.

## Início rápido (sem Docker)

```bash
git clone https://github.com/YuukiFST/Credit-Guard.git
cd Credit-Guard
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# Edita o .env: define GROQ_API_KEY se usares Groq
```

Treina uma vez se ainda não tiveres `models/best_model.pkl`:

```bash
python -m src.models.trainer
```

Sobe a API (usa outra porta se a 8000 estiver ocupada no Windows):

```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
```

Num segundo terminal, inicia o Streamlit. Se a API **não** estiver na porta 8000, define o URL base:

```bash
export CREDITGUARD_API_URL=http://127.0.0.1:8001   # macOS/Linux
# Windows PowerShell: $env:CREDITGUARD_API_URL = "http://127.0.0.1:8001"
python -m streamlit run app/streamlit_app.py
```

Com **Make** (se tiveres `make` e `uv`):

```bash
make dev
make train    # opcional
make api
make app      # noutro terminal
```

## Docker (opcional)

```bash
cp .env.example .env
docker compose up -d
```

Usa isto se preferires contentores em vez de um venv local.

## Como encaixa tudo

1. Carrega linhas no estilo UCI, aplica feature engineering, treina ou carrega o modelo serializado.
2. A API valida o JSON com Pydantic, corre o modelo e o SHAP e, se estiver configurado, chama o cliente LLM.
3. O Streamlit é um cliente fino: envia o mesmo formato JSON para `/predict` e `/explain`.

As decisões podem ser registadas em JSONL em `audit_trail/` (por defeito ignorado pelo Git).

## Estrutura do repositório

- `src/api` — aplicação FastAPI e rotas
- `src/audit` — registo de auditoria em JSONL
- `src/config.py` — configuração (YAML + ambiente)
- `src/data` — schemas Pydantic e carregamento de dados
- `src/explainability` — integração SHAP
- `src/features` — engenharia de atributos
- `src/llm` — cliente LLM e guardrails
- `src/models` — treino, métricas, calibração
- `app` — Streamlit
- `tests` — pytest

Para documentação de arquitetura mais longa, guarda uma cópia **fora** do repositório (por exemplo numa pasta `resources/` só local que não commits) ou estende este README no teu fork.

## Testes

```bash
pytest
```

O `pyproject.toml` define as opções de cobertura (incluindo o mínimo). Se `make test` falhar, corre `pytest` diretamente ou alinha o Makefile com o mesmo `--cov-fail-under` que está no `pyproject.toml`.

## Licença

MIT. Vê [LICENSE](LICENSE).
