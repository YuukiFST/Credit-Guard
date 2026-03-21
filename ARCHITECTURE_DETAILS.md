# 🏗️ Detalhamento Técnico e Arquitetural: Credit Guard

Este documento descreve a arquitetura de software, as decisões de design e o stack tecnológico subjacente ao **Credit Guard**. O foco desta documentação é prover contexto técnico a engenheiros de software, engenheiros de ML e DevOps mantenedores do sistema.

---

## 1. Visão Geral da Arquitetura (System Overview)

O repositório implementa um pipeline _end-to-end_ de Machine Learning com uma camada de inferência assíncrona baseada em LLMs. O design pattern primário segue uma arquitetura orientada a serviços (SOA) restrita a containers locais (Docker Compose), dividida estruturalmente em:

1. **Model Training Pipeline** (Offline / Batch)
2. **Inference Service REST API** (Online / Real-time)
3. **Client UI** (Frontend interativo)
4. **Experiment Tracking & Logging** (Observabilidade)

---

## 2. Stack Tecnológico: Bibliotecas & Frameworks

### 2.1. Machine Learning & Data Processing

- **Pandas (`pandas`)**: Utilizada para data wrangling, feature engineering e transformações vetorizadas. Escolhida pela integração nativa com o ecossistema Scikit-learn e facilidade de manipulação de features em memória.
- **Scikit-learn (`scikit-learn`)**: Fornece as abstrações primárias de pré-processamento (`StandardScaler`, `LabelEncoder`) e calibração de modelos (`CalibratedClassifierCV`). A calibração (Platt scaling / Isotonic regression) é crítica para garantir que as saídas do `predict_proba()` reflitam probabilidades reais.
- **XGBoost (`xgboost`) & LightGBM (`lightgbm`)**: O _core_ de predição do risco de crédito. Ambos implementam Gradient Boosting Frameworks em árvores de decisão.
  - _LightGBM_ utiliza crescimento de árvore _leaf-wise_, otimizando a velocidade de treinamento e consumo de memória estrutural.
  - _XGBoost_ atua como benchmark primário dado seu excelente tratamento intrínseco de dados tabulares não-lineares.
- **Optuna (`optuna`)**: Framework para _Hyperparameter Optimization_ (HPO). Ao invés de _Grid/Random Search_, implementa busca bayesiana (ex: amostrador TPE - Tree-structured Parzen Estimator), convergindo para hiperparâmetros ótimos em menos iterações computacionais.
- **Imbalanced-learn (`imbalanced-learn`)**: Fornece suporte ao método SMOTE (Synthetic Minority Over-sampling Technique) para interpolar novas amostras sintéticas da classe minoritária no espaço de features, mitigando viés (_bias_) algorítmico contra a classe de inadimplência frequente nos datasets financeiros.

### 2.2. Backend & Model Serving (API Layer)

- **FastAPI (`fastapi`)**: Framework assíncrono sobre ASGI (Uvicorn). Selecionado pela alta vazão de I/O em endpoints de inferência combinados (ML Model + External LLM API calls). Utiliza Pydantic nativamente para serialização.
- **Pydantic (`pydantic` v2)**: Responsável pelo parser e validação rigorosa dos payloads via schema estático. O núcleo reescrito em Rust _pydantic-core_ garante menor overhead de latência na validação cruzada das features de entrada da API, evitando _TypeErrors_ diretos nos modelos do scikit-learn.

### 2.3. Frontend Client

- **Streamlit (`streamlit`)**: Solução reativa baseada no padrão de injeção de estado. Evita a criação pontual de um frontend SPA estático (React/Vue/etc.), entregando componentes parametrizados para consumo imediato dos endpoints da API REST, acelerando prototipagem de fluxos do usuário.

### 2.4. DevOps & Infraestrutura

- **Docker & Docker Compose (`docker-compose.yml`)**: Isolamento do ambiente cross-platform e orquestração de microsserviços. Garante paridade entre os ambientes de desenvolvimento, _Continuous Integration_ e produção sem risco de colisão de pacotes nas variáveis de ambiente.
- **`uv` by Astral**: Resolvedor de dependências em Rust. Utilizado como drop-in replacement para o `pip` nos _Dockerfiles_ e CI/CD. Reduz o tempo de build em ambiente efêmero em ordem de magnitude.
- **MLflow (`mlflow`)**: Atua como o Model Registry e Tracking Server. Toda execução de hiperparâmetros pelo `optuna` registra suas dependências de baseline via `mlflow.log_metrics()` e salva os _pickles_ via `mlflow.sklearn.log_model()`. Consumível em `localhost:5000`.
- **GitHub Actions (`ci.yml`)**: Definição de integrações contínuas, garantindo que nenhum PR sofra _merge_ em `main` sem validação estática de tipos, _linting_ de estilo ou testes unitários com thresholds rigorosos.

### 2.5. Explicabilidade (XAI) & Prompt Engineering

- **SHAP (`shap`) - Shapley Additive exPlanations**: Extrai atribuição de features do modelo via Teoria dos Jogos. Fornece `TreeExplainer` para processar a importância local e global dos _gradient boosters_ (XGBoost/LightGBM) em $O(TLD^2)$. Responsável por gerar os tensores de atribuição ("vetores lógicos") da predição.
- **Groq API / Ollama (`ollama`, `groq`)**: A camada iterativa sobre o SHAP.
  - O JSON matemático gerado pelo SHAP não é _user-friendly_ nem cumpre métricas de compliance e LGPD na interface final (Auditoria Cidadã).
  - O backend aciona via _HTTP Client Assíncrono_ a requisição onde o modelo LLM decodifica como os shap values numéricos impactam o cenário em uma linguagem coesa para o tomador de decisão.

### 2.6. Developer Experience (DX) & Toolchain de Qualidade

- **Mypy (`mypy`)**: Checagem de tipagem estática mitigando os bugs comuns de tipagem fraca no Python. Todas as injeções e retornos da API estão submetidas à cláusula de `strict=true` (Mypy).
- **Ruff (`ruff`)**: Substitui _Flake8, Black_ e _isort_ com execução <10ms. Otimiza o bloqueio no CI de commits "sujos".
- **Pytest (`pytest`)**: Testes automatizados executando a API modularizada utilizando o _TestClient_ via injeção (`conftest.py`) e _mocking_ nas interações de LLM (testes paralelos independentes).

---

## 3. Fluxo de Vida de Execução (End-to-End Workflow)

1. **Treinamento e Versionamento:**
   O módulo `src/models/trainer.py` invoca data loaders e pipeline de pré-processamento. A otimização ocorre dentro da thread do _Optuna_. Ao fim da otimização, invoca-se `mlflow` para gravar artefatos em bucket estático (neste estágio, disco local `mlruns/`).
2. **Setup Pós-Deploy:**
   Ao rodar a composição em `docker-compose up`, o contêiner de API extrai o binário validado mais recente da persistência (ou aponta estaticamente ao modelo mapeado local na `/models`).
3. **Pico de Inferência HTTP:**
   1. Payload JSON serializado entra no Uvicorn (FastAPI Worker).
   2. _Pydantic_ valida _schemas_ e emite HTTP 422 em caso de falha sintática.
   3. O modelo aplica preditor probabilístico. Retorna-se o output puro.
   4. O módulo _SHAP_ é interpolado aos resultados preditivos sob as covariáveis injetadas pelo usuário, gerando arrays locais de influência.
   5. A camada do cliente de LLM interpola a requisição e traduz a atribuição de feature e a devolve via stream ou chunk final assíncrono.
4. **Log e Auditoria:** Registrado estruturalmente com o log (`logger.py` e logs imutáveis JSONL) mapeando IDs, payload raw, predição e _prompt output_ por razões estritas de auditoria (Compliance / Explainable AI).
