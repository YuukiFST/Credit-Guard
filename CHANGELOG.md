# Changelog

Todas as mudanças notáveis neste projeto serão documentadas aqui.
O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

## [1.0.0] - 2026-XX-XX

### Adicionado
- Pipeline completo ML + GenAI para análise de risco de crédito
- Comparativo de 3 modelos (RF, XGBoost, LightGBM) com Optuna
- Calibração de probabilidades com CalibratedClassifierCV
- API REST com FastAPI (endpoints /predict, /explain, /health)
- Fallback LLM: Ollama local → Groq API
- Guardrails de validação pós-geração do LLM
- Explicabilidade via SHAP com narrativas em linguagem natural
- Análise de viés/fairness por grupo demográfico
- Feature selection baseada em SHAP + correlação
- Testes automatizados com cobertura > 75%
- Docker Compose com 4 serviços (API, Streamlit, MLflow, Ollama)
- CI/CD com GitHub Actions (lint + testes + validação de modelo)
- Audit trail em JSONL para compliance LGPD
- Interface Streamlit interativa com ajuste de threshold

### Detalhes Técnicos
- Feature engineering: 8+ features derivadas
- Threshold dinâmico via análise custo-benefício
- Validação de dados com Pydantic v2
- Logging estruturado com Loguru
- Makefile para DX (Developer Experience)
- Monitoramento de drift via PSI
