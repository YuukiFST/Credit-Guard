# 💳 Credit Guard — ML + GenAI para Análise de Risco de Crédito

> Sistema end-to-end de predição de inadimplência combinando XGBoost,
> explicabilidade via SHAP e narrativas em linguagem natural via LLM.
> Construído com boas práticas de engenharia de software: testes, CI/CD,
> Docker e API REST.

<!-- Badges -->

![CI](https://github.com/seu-usuario/CreditGuard/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- GIF da interface em ação -->
<!-- ![Demo](docs/assets/demo.gif) -->

## 🚀 Demonstração Rápida

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/CreditGuard.git
cd CreditGuard

# Opção 1: Docker Compose (recomendado)
cp .env.example .env
docker-compose up -d

# Opção 2: Execução local
make dev          # instala dependências + hooks
make train        # treina modelos (RF, XGBoost, LightGBM)
make api          # inicia API em localhost:8000
make app          # inicia Streamlit em localhost:8501
```

## 📊 Resultados

> _As métricas de modelo serão adicionadas após o treinamento do modelo final com Optuna._\_

## 🏗️ Arquitetura

```
📊 Dataset UCI          🧠 ML Pipeline           🌐 API REST          🖥️ Frontend
Credit Card  ──→  Feature Engineering ──→  FastAPI /predict  ──→  Streamlit
                  SMOTE + XGBoost            └─ /explain
                  Calibração                    └─ /health
                  SHAP                            │
                                          🤖 LLM (Ollama→Groq)
                                          📝 Audit Trail JSONL
```

## 🎯 Diferenciais vs Projetos Comuns

| Aspecto        | Projeto Típico                | Credit Guard                           |
| -------------- | ----------------------------- | -------------------------------------- |
| Pacotes        | `pip` (Lento, síncrono)       | `uv` (Rust-based, 10-100x mais rápido) |
| Modelos        | 1 modelo (RF), sem comparação | 3 modelos comparados com Optuna        |
| Threshold      | Fixo em 0.5                   | Otimizado por custo-benefício          |
| Probabilidades | Raw predict_proba             | Calibradas (CalibratedClassifierCV)    |
| Validação      | Sem schema                    | Pydantic v2 completo                   |
| API            | Apenas Streamlit              | FastAPI + Streamlit                    |
| LLM            | Só Ollama local               | Fallback: Ollama → Groq                |
| Guardrails     | Nenhum                        | Validação pós-geração                  |
| Fairness       | Não analisado                 | Disparate Impact Ratio                 |
| Testes         | Zero                          | >75% cobertura                         |
| Drift          | Não monitorado                | PSI por feature                        |
| Deploy         | Apenas local                  | Docker + CI/CD                         |

## 📁 Estrutura do Projeto

```
CreditGuard/
├── src/
│   ├── config.py              # configuração central
│   ├── data/                  # schemas Pydantic + loader
│   ├── features/              # feature engineering + seleção
│   ├── models/                # trainer, evaluator, calibrator, fairness, monitoring
│   ├── explainability/        # SHAP explainer
│   ├── llm/                   # client com fallback + guardrails
│   ├── audit/                 # log imutável de decisões
│   └── api/                   # FastAPI + routers
├── app/                       # frontend Streamlit
├── tests/                     # unit + integration
├── notebooks/                 # EDA + model comparison
├── Dockerfile                 # API
├── Dockerfile.streamlit       # Streamlit
├── docker-compose.yml         # 4 serviços
├── Makefile                   # DX
└── .github/workflows/ci.yml   # CI/CD
```

## 🧪 Testes

```bash
# Rodar todos os testes com cobertura
make test

# Ou diretamente:
pytest --cov=src --cov-report=html
```

## 🔧 Tecnologias

| Camada          | Tecnologia                              |
| --------------- | --------------------------------------- |
| ML              | scikit-learn, XGBoost, LightGBM, Optuna |
| Explicabilidade | SHAP                                    |
| LLM             | Ollama, Groq API                        |
| API             | FastAPI, Pydantic v2                    |
| Frontend        | Streamlit, Plotly                       |
| Qualidade       | Ruff, mypy, pytest                      |
| Infra           | Docker, GitHub Actions, MLflow, uv      |
| Dados           | Pandas, imbalanced-learn (SMOTE)        |

## 📚 Referências

- [Dataset: UCI Credit Card](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [SHAP Library](https://shap.readthedocs.io/)
- [LGPD Art. 20](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
