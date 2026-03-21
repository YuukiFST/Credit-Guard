# PROJECT_CONTEXT.md

## 1. VISÃO GERAL TÉCNICA

O **Credit Guard** é um sistema de avaliação de risco de crédito com **Machine Learning** supervisionado (para classificação de default) e explicabilidade injetada via **Large Language Models (LLM)**. Seu propósito é ingerir dados financeiros de um cliente (limite, histórico, idade), prever a probabilidade de inadimplência (classificação binária), extrair os fatores que mais pesaram na decisão e traduzi-los de matemática para linguagem natural.

### Processos e Serviços

O sistema orquestra 4 serviços em rede via Docker Compose:

1. **API Backend (FastAPI)** (`:8000`): Controlador central, expõe endpoints, valida dados, coordena inferências de ML e chamadas assíncronas a LLM.
2. **Frontend (Streamlit)** (`:8501`): Interface web do usuário, captura variáveis de input e renderiza outputs (gráficos e texto) acessando a API vis REST.
3. **Tracking de Experimentos (MLflow)** (`:5000`): Servidor de backend com banco SQLite local (`mlflow.db`) para registro de hiperparâmetros, métricas e artefatos de modelo durante o treinamento.
4. **Daemon Local de IA (Ollama)** (`:11434`): Servidor local de LLM que gerencia inferências na máquina sem depender de internet, rodando a imagem padrão.

### Protocolos e Formatos

- A comunicação entre o _Frontend_ e a _API_ ocorre estritamente via **HTTP POST** trafegando **JSON** no Body. A API usa o formato `application/json`.
- A API retorna dados tipificados por schemas definidos e HTTP status codes nativos (`200 OK`, `422 Unprocessable Entity` para erros de tipo, `500 Internal Server Error`).

```text
[Usuário] --> browser --> [Streamlit :8501]
                                |
                                | (HTTP POST JSON) -> /predict, /explain
                                v
                      [FastAPI Server :8000]
                      /                    \
     (Local HTTP)    /                      \ (External HTTP/TCP)
                    v                        v
          [Ollama :11434]               [Groq API Cloud]
```

---

## 2. ESTRUTURA DE ARQUIVOS

### Raiz do Projeto

- `pyproject.toml`: Configuração central do projeto (meta, toolings como ruff, mypy, pytest e coverage).
- `config.yaml`: Centraliza todas as configurações declarativas do sistema que não são sensíveis (paths, hyperparâmetros, nomes de modelos).
- `.env.example` / `.env`: Armazena _apenas_ dados injetivos e sensíveis como a variável obrigatória `GROQ_API_KEY`.
- `Makefile`: Hub abstrato de comandos CLI contendo rotinas de `dev`, `lint`, `test`, `api`, `app`, `docker-up`, `train`. Responsável por ser o orquestrador padrão das mãos do dev.
- `docker-compose.yml`: Orquestração global que interliga todos os serviços de aplicação (API, app web, MLflow, Ollama), expõe as portas e configura diretórios de volumes locais (_bind mounts_).
- `Dockerfile`: Script de compilação imutável para a imagem do Backend em Python 3.12-slim, instalando requisitos de sistema e empacotando o subdiretório `src`.
- `Dockerfile.streamlit`: Script atrelado ao Streamlit que copia a subcamada `app/` e exporta na porta 8501.

### Diretório `app/` (Frontend)

- `streamlit_app.py`: Interface construída em Streamlit. Responsável apenas por desenhar a tela, capturar os formulários estritos e efetuar HTTP streams de `POST` para `localhost:8000`.

### Diretório `src/` (Aplicação Core Backend e ML)

#### `src/config.py`

Ponto de injeção global. Aglutina `config.yaml` e as chaves `.env` num objeto estrito de `Pydantic BaseSettings`. Todo arquivo subsequente importa `from src.config import settings`.

#### `src/api/` (Borda da Rede)

- `main.py`: Ponto de entrada Uvicorn para o app web distribuído. Inicializa configuração, setup de rotas e ciclo de vida (`lifespan`).
- `routers/health.py`: Contém `/health` (`GET`) retornando `{status: ok}`, utilitário vital para _Docker Healthcheck_.
- `routers/predict.py`: Contém `/predict` (`POST`). Coleta payloads restritos e empurra no Pipeline de inferência Machine learning (Semânticamente isolado). Traz acoplamento com o AuditLogger e SHAP.
- `routers/explain.py`: Contém `/explain` (`POST`). Invoca internamente ML+SHAP e delega o output vetorial traduzido num prompt textual para infraestrutura Async LLM (Ollama/Groq).

#### `src/data/` (Contratos e Ingestão)

- `schemas.py`: Validador de I/O em runtime, definindo objetos `CreditClientInput` e `PredictionOutput` pelo Pydantic, garantindo ranges estritos de números e conversão fluida para formato tabular pandas (`to_dataframe()`).
- `loader.py`: Subsistema que aponta via disco a ingestão dos dados crús tabulares (ex. `UCI_Credit_Card.csv`) e fraciona em Train/Test estratificados, alimentando as subcamadas modeladoras.

#### `src/features/` (Álgebra Multidimensional)

- `engineer.py`: Classe modular (Pipeline de Feature Engineering _Stateless_) que transforma arrays bidimensionais através de proporções vetoriais, eliminando IDs e calculando Credit Utilization e Payment Delay sem Loops python puros (altamente performático).
- `selector.py`: Executa processamento baseado em análise global das contribuições Shapley e correlação de Person `corr_matrix` das features geradas, a fim de extrair subconjuntos imensamente colineares/desimportantes do processamento e reduzir a carga das features.

#### `src/models/` (ML Training & Inference)

- `trainer.py`: Motor Cérebro da Inteligência de dados. Otimiza XGBoost, LightGBM, Random Forest baseados na métrica ROC_AUC usando TPE do Optuna. Armazena no MLflow tracking e finaliza serializando o Joblib em `best_model.pkl`. Usado apenas em script offline, não é carregado em cache runtime a não ser na predição via API.
- `evaluator.py`: Componentiza as sub-avaliações. Contém `calculate_full_metrics` (ROC_AUC, Gini, F1, KS_Statistic) e função de simulação matemática em `find_optimal_threshold` que mescla falsos positivos contra métricas reais de Business.
- `calibrator.py`: Subsistema para encurtar variabilidade real de Probabilidade vs Incidência utilizando calibração isotônica e scaling scikit-learn.
- `fairness.py`: Implementação LGPD estrita analisando e mitigando o Disparate Impact Ratio contra viés predefinido sobre Atributos Demográficos do modelo.
- `monitoring.py`: Script logaritmico de Drift detecção calculando índices como o Populational Stability Index (PSI). Detecta mudanças drásticas pós-produção na amostragem.

#### `src/explainability/` (Transparência de Grafo)

- `shap_explainer.py`: Empresta a subestrutura TreeExplainer implementada em `C++`, processando vetores SHAP pra inferir quem cooperou marginal a predição local do algoritmo "caixa-preta".

#### `src/llm/` (Geração e Fallback)

- `client.py`: Arquitetura paralela englobando clientes `Ollama` e `Groq`. Cria orquestração de resseguro via Fail-Over assincrono em httpx para sempre processar um retorno textual, bloqueando quedas caso a rede esteja caída ou localmente engolada.
- `guardrails.py`: Regras estritas de parser no Regex final proibindo palavras ilegais, descriminatorias, vazamentos (_garanto aprovacao_, PII etc). Funciona de _Sanitizer_/Filtro limpo do GenAI.
- `prompts/`: Contém templates restritos que alimentam contextos injetáveis no encoder via Text Files puros como `v2_system.txt`.

---

## 3. STACK TECNOLÓGICO COMPLETO

| Tecnologia           | Versão       | Papel Específico no Projeto                                           | Localização                                  | Configuração Adicional                                              |
| -------------------- | ------------ | --------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------- |
| **Python**           | 3.11+        | Linguagem host. Orquestra as instâncias numéricas por cpython         | Global                                       | Tipagem forte `mypy` habilitada `strict=true`                       |
| **FastAPI**          | _padrão req_ | Roteador REST assíncrono sobre um EventLoop UVLoop local              | `src/api/*`                                  | Registrado com metadados de Lifecycle `lifespan`                    |
| **Pydantic**         | _v2 nativa_  | Parser/Coercer C++/Rust de validação em Tempo Real na Borda           | `src/data/schemas.py`, config `BaseSettings` | Configurado com extra="ignore" no settings load                     |
| **Streamlit**        | _padrão req_ | Renderização UI server-driven HTML isolada fora da responsividade     | `app/streamlit_app.py`                       | Roda na porta 8501 e é injetada css inline                          |
| **Pandas/Numpy**     | _padrão req_ | Estrutura de memória `C-contiguous` pro ML, álgebra vetorial          | `src/features/*`, loader, ML                 | Evita loop For em processamento                                     |
| **Scikit-Learn**     | _padrão req_ | Bibliotecas utilitárias CrossVal, StratifiedFold, Calibrator          | `src/models/*`                               | Usado para pipelines base de pre-processamento                      |
| **Optuna**           | _padrão req_ | Maximização do processo Bayesiano em hyperparametros TPE Algoritmo    | `src/models/trainer.py`                      | Busca de n_trials setado iterativamente no YAML                     |
| **XGBoost/LightGBM** | _padrão req_ | Motores de Estimação Não-Paramétrica em Árvores em C++ / OpenMP       | `src/models/trainer.py`                      | Utilizados com instâncias n_jobs=-1 para paralelismo extremo na CPU |
| **SHAP**             | _padrão req_ | Transparência computacional via Teoria dos Jogos Cooperativos Game    | `src/explainability/*`                       | Implementa classe restrita `TreeExplainer(model)`                   |
| **MLflow**           | 2.14.1       | Servidor/Repositório de dados e métricas do pipeline (`.db` e `.pkl`) | `src/models/trainer.py`                      | Subido em porta isolada Docker 5000 apontando ao dir `mlruns/`      |
| **Ollama/Groq API**  | -            | Processamento LLM em fallbacks restritivo do modelo gerativo text     | `src/llm/client.py`                          | Modelos default: phi3:mini (local) e Llama 3.1 8B (nuvem)           |

---

## 4. ARQUITETURA E FLUXO DE DADOS

### 4.1. Componentes e Responsabilidades

1. **API Router (`Predict`)**: Invoca validação strict (`CreditClientInput`), empurra na Classe Engineer (`FeatureEngineer`), instancida Inferência do Modelo (`best_model.pkl` cacheado), aciona Graph Explanation em C++ (`SHAPExplainer`), Audit Logs result em JSON, Output Client Data via Retorno 200 JSON OK.
2. **API Router (`Explain`)**: Invoca Predict Pipeline supra citado e Acopla I/O Subsystem (`LLMClientWithFallback`) aguardando Asincronamente e Validando a Resposta de volta a Client.
3. **Streamlit App**: Controla Layout, Acopla Sockets no host HTTP (porta:8000), extrai e renderiza grafos Plotly (Bar Chart de Fatores).
4. **Trainer Engine**: Componente Desktop Offline (Síncrono/Bloqueante) de rodízio e computação de pesos. Invoca Load_Data, Prepara Feature Engine, Realiza K-Fold Optuna, sube Modelos pro HD físico, envia Tracking API HTTP para MLflow Server restrito, Serializa a RAM via Joblib e Salva (`best_model.pkl`).

### 4.2. Fluxo Completo de uma Requisição (/explain)

1. **Ponto de entrada:** Usuário clica "Analisar Risco" na Interface. Streamlit formata o Dicionário.
2. HTTP POST enviado para FastAPI `127.0.0.1:8000/explain`.
3. Pydantic engole Payload de Bytes, instanciando Class obj estrito `CreditClientInput`. Falha se `age="Quarenta"` com HTTP 422.
4. FastAPI delega para rota `async def explain_decision()`.
5. Imports lazy ocorrem, instanciando Singleton do `model` da ram e do `FeatureEngineer`.
6. Dados transpostos para Array Numpy interno do Pandas via `to_dataframe()`. Matriz é mutacionada Algebraicamente (`fe.transform`).
7. Executado método `predict_proba` síncrono da instância arbórea sobre a Matriz retornando Escalar (Float ex: 0.85). Limpo.
8. SHAP Engine acionada, percorre Graph Arvores extraindo array NumbPy de valores com 5 contribuições maiores via Sorting Logarítmico.
9. Roteador forma o _Prompt Final_ com os Escalares (Decisão, Fatores SHAP). Chama função Asincrona de I/O em backend Groq/Ollama de GenAI (`await llm.generate(...)`). FastAPI "dorme" liberando CPU.
10. Sockets IO respondem os bytes em Stream gerativo, e função é reativada, varrendo a resposta gerada com GuardRails (`safe_generate`).
11. Roteador empacota Output model JSON. API transmite via Socket ao browser / App.

### 4.3. Fluxo de Erros

- Erro **De Formatação Típica (Validation)**: Tratado na Raiz HTTP middleware com Exceção `RequestValidationError` → devolve Code 422 formatado para User Error nativo do Fastapi.
- Erro **Indisponibilidade API Externa (Network I/O)**: O `llm/client.py` absorve Exceção `httpx.ConnectError` sob Bloco Except abrangente do Ollama, Transicionando percurso sequencial para Instalar em Client do Groq. Caso Falhe novamente a Engine retorna mensagem "Hard-Coded Default", nunca expondo `500 Server Error` à API HTTP se o núcleo essencial de probabilidade funcionar.
- Erro **Genérico do Servidor (Crash Interno)**: A rota `predicy.py` encapsula a Lógica core em Bloco `Try / Except Exception as e` abrangente gravando rastros logísticos e formatando retransmissão controlada em HTTPException Code `500 Internal Server Error`, impedindo vazamento de Stack Trace subjacentes que possam transbordar na rede REST.

---

## 5. CONTRATOS DE DADOS

### `CreditClientInput`

A matriz de predições aceitas no Input.

- `limit_balance` (float, gt=0): NT$. Padrão sem validation, falha se <= 0.
- `age` (int, 18-100): Anos.
- `sex` (int, 1 ou 2): 1=M, 2=F.
- `education` (int, 1-6): Graduação até outros.
- `marriage` (int, 0-3): Estado.
- `pay_history` (list[int], min/max len=6): Arrays passadas contendo Valores (-2 a 9). Possui validador Customizado via Classe bloqueando números absurdos.
- `bill_amounts` (list[float], len=6) e `pay_amounts` (list[float], len=6): Valores faturados/pagos NT$. Validador Customizado modela Warning se payment excede duas vezes a Fatura sistematicamente em 3+ meses.
- Produzido: Pela UI em `app/streamlit_app.py` num JSON body.
- Consumido: Pelo FastAPI injetado na Rota em `src/api/routers/[predict|explain].py`.

### `PredictionOutput`

Resposta da Rota finalizada.

- `client_id` (str): UUIDv4 autogerado internamente no endpoint para Idempotência.
- `default_probability` (float 0 a 1): Calculado pela inferência do Grafo de ML.
- `decision` (str): "APROVADO" ou "NEGADO".
- `threshold_used` (float): Configurado vs Probabilidade Real.
- `top_factors` (list[dict["feature": str, "shap_value": float]]): Dicionarios processados via explainer Numpy class.
- `model_version` (str): string.
- `timestamp` (str): Horário Isoformat.
- Produzido: Pela Rota Predict na Retornada.
- Consumido: Pelo Streamlit em parse Json após a res.status_code == 200.

---

## 6. MODELOS DE MACHINE LEARNING

- **Algoritmos**: O sistema testa competidores por Boosting Paralelos e Seq: RandomForest, XGBoost, e LightGBM.
- **Treinamento**: Efetuado via script `python -m src.models.trainer`.
  1. Carrega UCI dataset e divide via StratifiedKFold (0.2 em test size), seed constante=42.
  2. Submete matriz com Features Engineered pra rodadas iterativas bayesianas no OPTUNA n_trials (ex 50 times). Tuning massivo de Estimadores, learning rate, profundidade do grafo (`max_depth`). A Função Objetivo perpassada por Optuna busca maximizar Métrica do ROC_AUC na base cruza (CV=5 cross validation scoring).
  3. Exporta log via Mlflow na rede Local da máquina hospedeira.
  4. Extrai O vencedor Global, calibra, e instancia via JobLib o artefato Byte `models/best_model.pkl`.
- **Carregamento Memória Produção**: API REST via função `load_model()` importa Joblib na ram uma única fez na importação da rota tardia do Python `lazy importing global variables`.
- **Inputs ao predict_proba()**: A matriz final transposta possui as colunas bases do Cliente, MENOS Colunas Targets/IDs, MAIS features aglutinadas geradas como `CREDIT_UTILIZATION` e `MONTHS_WITH_DELAY`. O modelo avalia matriz nD do Numpy.
- **Outputs predict_proba()**: Modelo exporta Numpy Array Flutuante das densidades nas classsification 0 vs 1. API captura fatia `[:, 1][0]` equivalendo a Probalidade Relativa contínua do Risco/Defaulter (Ex. 0.81). Subseqüentemente transposto ao _Threshold_ resultando em Classificatoria (Negado).

---

## 7. INTEGRAÇÃO COM LLM

- **Orquestrador LLM**: Reside em `src/llm/client.py`. Baseia-se em Instabilidade Ativa de Arquitetura Subjacente LLM Client With FallBack.
- **Provedor Primário (Local, Privado)**: Instância nativa Daemon `Ollama`. Modelo ativado `phi3:mini`. Timeout estipulado em URL `11434`. Latência nula exterior. Custo zero.
- **Provedor Secundário (Nuvem, Público)**: Acionado mediante Timeout do Primário O.S. API REST remota provedora Groq acessada com EndPoint em `https://api.groq.com/openai/v1/chat/completions`. Requer Secret Auth injetado via `.env` Headers `Bearer`. Modelo parametrizado `llama-3.1-8b-instant`.
- **Prompt Input**: Texto unificado, compõe Strings Interpoladas dos Fatores SHAP limitados via função `_build_explain_prompt` formatando Bullet points num Markdown Interno do prompt. A mensagem de Systema de Contexto restringe a atitude via arquivos em raiz `src/llm/prompts/`. Submetido via Array of Dicts contendo "Role", "Content". Configurado Temperature nula (Estável/Não Criativo). Max Predict 500 Tokens.
- **Processamento/Resposta**: HTTPx assíncrono consome API e aciona o Parser guardrail. A Classe Guardrail verifica substrings proibitiva (como promessas falsificadas) impedindo a visualização da Resposta caso a IA "Alucine".
- **Tratamento de Falha Definitiva**: Na impossibilidade remota Global Sub-Redes indisponíveis de Nuvem Groq e Ollama mortos: Código processador resgata Catch `Exception` Exato sem propagar. Assuma-se retorno hardcoded Padrão (_System Default String_) mantendo sistema operante na Rota 200 HTTP Rest.

---

## 8. CONFIGURAÇÃO E VARIÁVEIS DE AMBIENTE

As configurações são distribuídas entre variáveis de ambiente (sob `.env`) e arquivos (`config.yaml`). O Pydantic consolida as fontes em `Settings` em Runtime.

| Variável Config           | Arquivo de origem | Tipo    | Valor Padrão (YAML)            | Obrigatória            | Efeito no Sistema                                                                                        |
| ------------------------- | ----------------- | ------- | ------------------------------ | ---------------------- | -------------------------------------------------------------------------------------------------------- |
| `GROQ_API_KEY`            | `.env`            | string  | `""`                           | SIM (Se fallback usar) | Autentica a API Groq. Se não instanciado o fallback do LLM tranca em restrição Http 401.                 |
| `ENVIRONMENT`             | `.env`            | string  | `development`                  | NÃO                    | Modulador base pra Logs e sub-deploy flag.                                                               |
| `data_raw_path`           | `config.yaml`     | Path    | `"data/raw/UCI_...csv"`        | NÃO                    | Define matriz lida na ingestão offline.                                                                  |
| `data_target_column`      | `config.yaml`     | str     | `"default.payment.next.month"` | NÃO                    | Coluna Target pro processamento da matriz Y True em treino.                                              |
| `data_test_size`          | `config.yaml`     | float   | `0.2`                          | NÃO                    | Proporção retenção validação de dados cross-validation.                                                  |
| `model_default_threshold` | `config.yaml`     | float   | `0.5`                          | NÃO                    | Limiar corte binário base (0.5 ou maior acorda Negar crédito). Podendo ser sobscrita pela GUI Streamlit. |
| `model_cv_folds`          | `config.yaml`     | int     | `5`                            | NÃO                    | Camadas de processamento em dobra Optuna K-folds Stratified.                                             |
| `model_optuna_trials`     | `config.yaml`     | int     | `50`                           | NÃO                    | Iterações para processamento exaustivo Bayesiano de Hyperparametros.                                     |
| `business_cost_fp`        | `config.yaml`     | float   | `500`                          | NÃO                    | Base matricial Falso Positivo pro cálculo Limiar de Otimização Business em `evaluator.py`.               |
| `business_cost_fn`        | `config.yaml`     | float   | `2000`                         | NÃO                    | Mesma regra acima, para Inadimplementos ignorados.                                                       |
| `llm_local_model`         | `config.yaml`     | string  | `"phi3:mini"`                  | NÃO                    | String para apontar ao Ollama run base o LLM parametrizado no pull container.                            |
| `llm_fallback_model`      | `config.yaml`     | string  | `"llama-3.1-8b-instant"`       | NÃO                    | String para chamada da API via Post HTTPS Grok na falência.                                              |
| `api_host` / `api_port`   | `config.yaml`     | str/int | `"0.0.0.0" : 8000`             | NÃO                    | Porta subida do ASGI Event Loop do Uvicorn para bind TCP.                                                |

---

## 9. DEPENDÊNCIAS COMPLETAS

Extraídas a partir das bases e declarações nos arquivos. Compõe ecossistema isolado de sub-layers CPython.

| Pacote                              | Versão Específica        | Usado em (arquivo)                        | Função específica no projeto                                                                               |
| ----------------------------------- | ------------------------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **fastapi**                         | -                        | `src/api/` (main/routers)                 | Roteador e servidor rest ASGI framework de requisições de borda Assícrono                                  |
| **uvicorn**                         | -                        | `Dockerfile`, `Makefile`                  | Servidor TCP Event Loop worker runner operando sob fastapi                                                 |
| **uv (Astral)**                     | _nativo docker/Makefile_ | Instalações e resolução de pacotes Python | Substituto moderno e veloz em Rust para o `pip`. Acelera o setup de ambiente e CI/CD.                      |
| **pydantic**, **pydantic-settings** | -                        | `src/data/schemas.py`, `src/config.py`    | Engine parser validador coercitivo Rust/python de runtime Tipagem e Envs                                   |
| **streamlit**                       | -                        | `app/streamlit_app.py`                    | Framework Backend-For-Frontend gerando o SPA renderizador HTML/CSS/JS por scripts py                       |
| **pandas**, **numpy**               | -                        | `features/engineer`, ML e data            | Álgebra de sub-matrizes bidimensionais com processadores alocados sub-CPython via vetorização C-Contíguous |
| **scikit-learn**                    | -                        | `models/trainer`, `data/loader`           | Funções e heurísticas ML Base: Scalers, StratifiedKFold, CrossValidationScore                              |
| **xgboost**, **lightgbm**           | -                        | `models/trainer.py`                       | Bibliotecas nativas C++ de Gradient Boosting Trees Ensemble algoritmos primários                           |
| **optuna**                          | -                        | `models/trainer.py`                       | Mecanismo de sub-pesquisa Bayesiana Hyperparamétrica TPE Searcher interativa                               |
| **shap**                            | -                        | `explainability/shap_explainer.py`        | Computador polinomial logarítmico para cálculo matricial Shapley Values C++ Extractor individual predição  |
| **httpx**                           | -                        | `llm/client.py`, `app/streamlit_app.py`   | Cliente HTTP Asynchronous TCP nativos com suporte non-blocking threads asyncio a Requisições Web           |
| **mlflow**                          | `<2.14.1 via Compose>`   | `models/trainer.py`                       | Base logger e tracker da vida e métricas offline para comparação heurística parametrizada Treino           |
| **joblib**                          | -                        | `models/trainer.py`                       | Dumper de RAM C++ Object Pickle Serializer do Model Scikit                                                 |
| **loguru**                          | -                        | _(Em todos os routers)_                   | Gerador log nativo custom com trace multi-threading lock safe substituto via root logging file             |
| **pytest**, **ruff**, **mypy**      | -                        | CI / `pyproject.toml`                     | Mecanismos de Testing Pytest, Linter PEP8 Rust base e Static Type Checker estriict                         |

---

## 10. INFRAESTRUTURA E DEPLOY

O sistema gerencia orquestração inteiramente em `docker-compose.yml`:

1. **api**: Utiliza `Dockerfile`. Instala O Gerenciador `uv` ultra-rápido, copia camadas `requirements` separadas de sources script para maximizar caching de Docker layer, mapeia Variáveis, Volume e expõe :8000. Conta com Dependência de Health check curl e `depends_on` (Esperar ollama service alive). Sobe servido host local usando entry point CMD instanciando app API Web.
2. **streamlit**: Utiliza `Dockerfile.streamlit`. Roda porta `:8501`. Aguarda dependência sadia `condition: service_healthy` do `api` a fim de não apresentar timeout na Tela principal de start.
3. **mlflow**: Faz proxy na imagem pronta GHCR `mlflow:v2.14.1`. Comando embutido inicia Servidor hospedando no banco `SQLite` backend na porta `5000:5000`. Requer subpasta alocada via Volume `mlruns:` no Host para o BD durar entre execuções docker em queda subjacente.
4. **ollama**: Puxa O contêiner de daemon C++ nativo emulado linux na `ollama/ollama:latest` Porta 11434. Tem volume local `ollama_data` pra isolar os 2Gb dos pesos pesados .bin (Tensors model Phi3 persistentes e impedir downloads no boot).
   Todos interligados na Sub-Rede Default isolada do Compose habilitando requests via DNS nativo entre eles ex: `http://api:8000` via Container hostname.

---

## 11. COMANDOS E SCRIPTS OPERACIONAIS

Centralizado no Arquivo GNU `Makefile`:

- `make install`: Orquestra subrotina executorial CLI de Empacotador (`uv pip install -r requirements.txt`) operando no ambiente do Python em modo global.
- `make dev`: Preenche as pastas isolando testes, Liners (`requirements-dev.txt`) acopla os hooks locales precommit pra forçar checagens nos commits branch.
- `make lint` / `make typecheck`: Inicializa `ruff` C-based Linters formartador fix/pep8 e validações em código nativos, e Ramo de Validação Mypy subestrutural de Dictionaries na restrita conformação da Tipagem explícita.
- `make test`: Invoca engine _Pytest_, restrito a coverage test acima do escopo limite default `--cov-fail-under=75`.
- `make api` / `make app`: CLI local puro sem contêiner, subindo subprocesso `uvicorn` Hot Reload ativado `0.0.0.0` e Servidor local processado Python no browser via streamilit.
- `make train`: Ponto estrito e crítico da camada de treinamento. Delega ao Processo de Background executório CLI do modulo `python -m src.models.trainer`, criando Joblibs offline instanciados após KFold optuna local e encerra a linha O.S sem prender a Shell.
- `make docker-up` / `docker-down`: Atalho simplista do `docker-compose up -d --build`. Destrói tudo (Remove virtual containers na Subrede O.S isolada).

---

## 12. ESTADO DO PROJETO E DECISÕES QUE IMPACTAM O FUTURO

**O que está finalizado / operante**:

- Todo o processamento de predição via árvores de ramificação paralela em Numpy/Scikit.
- A orquestração das APIs web interconectadas ao frontend através do Docker com redes montadas.
- O subsistema completo de monitoração paralela em métricas SHAP explicáveis no contexto transacionado LLM client.

**O que aparenta estar Incompleto / Limitações Técnicas Analisadas**:

- Em `app/streamlit_app.py`, o LLM Client envia as respostas via timeout `timeout=60` em chamadas HTTP REST normais HTTPX. Uma execução demorada gerará loading lock completo em Streamlit sem WebSockets `Streaming/Chunking` progressivo das sílabas na tela. Em atualizações futuras, seria prudente adaptar os endpoints do fastAPI pra instanciar de Generator com `StreamingResponse` para que o frontend popule textos asincronos conforme os tokens de respostas chegam O.S.
- A Interface de Audit (`AuditLogger`) é acoplada assincronamente (e citada em predições como um script abstrato ativado para gravação da ID req e dados prob) no EndPoint /predict, contudo as interfaces tabulares da gravação são de Disco JSON solados - para Escala enterprise um SQL DB assincrono seria implementado na orquestração Docker para logs.
- Em termos arquiteturais de Machine Learning: Features estritas mapeadas no Numpy subjacente referendam O Pipeline UCI original Taiwan Credit base; caso novos formulários ingressem do frontend com nomes de matriz difereciados (`e.g cpf/endereco`), toda a sub-matriz e pipeline refatorar-se-iam forçosamente.

O documento final se encerra. A transposição arquitetural técnica deste cenário foi totalmente efetuada em 360-graus. A partir daqui LLMs e Agentes podem reconstruir, auditar, e reparar a estrutura integral.
