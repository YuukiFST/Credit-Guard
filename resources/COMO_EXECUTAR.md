# Guia Prático: Como Rodar e Testar o Credit Guard 🚀

Este documento é um passo a passo objetivo para subir todo o ambiente de Risco de Crédito (API + Frontend + MLflow + Ia Local/Cloud) e realizar seus testes práticos.

O projeto foi modernizado para ter a curva de execução mais rápida possível. Você tem duas opções: **Via Docker (Recomendado)** ou **Local (Modo Developer)**.

---

## 1. Passos Iniciais (Aplica-se a ambas as formas)

Antes de executar, você precisa configurar as chaves de fallback para a IA e ter o código clonado/disponível:

1. Renomeie (ou copie) o arquivo `.env.example` para `.env` na raiz do projeto.
2. Acesse sua conta no Groq ([console.groq.com](https://console.groq.com/)) ou use a chave temporária listada no arquivo se aplicável.
3. Preencha a variável no `.env`:
   ```bash
   GROQ_API_KEY=sua_chave_groq_aqui
   ```

_(Nota: O Groq é essencial como "fallback". Se a sua IA local via Ollama demorar muito a responder, o painel cai silenciosamente para a nuvem da Groq via API)._

---

## 2. Opção A: Execução via Docker (Maneira mais fácil) ✅

O modo via [Docker](https://www.docker.com/) é altamente sugerido pois já faz o setup da IA do banco de dados (MLFlow) sem sujar sua máquina.

### Pré-requisitos

- **Docker** e **Docker Compose** instalados na sua máquina.

### Como rodar

1. Abra um terminal na pasta raiz do projeto.
2. Execute o comando no Makefile ou o comando nativo do compose:

   ```bash
   # Opção usando Make:
   make docker-up

   # Ou usando o docker nativo:
   docker-compose up -d --build
   ```

Isso fará com que o sistema baixe:

- O banco local do MLFlow Tracker
- A infraestrutura da IA base (Ollama Daemon)
- Monte e compile o Streamlit UI e FastAPI (Backend)

_(Aguarde o processo de build do contêiner do Python)._

---

## 3. Opção B: Execução Local Completa (Modo Developer) ⚙️

Se deseja debugar o código fonte, ou não quiser o peso do Docker, siga esta instrução.

### Pré-requisitos

- Python instalado (`>= 3.11`).
- [Ollama instalado nativamente](https://ollama.com/) se desejar explicações locais de Machine Learning. _(Se optar por não instalar o Ollama, certifique-se de configurar a GROQ_API_KEY no .env para o sistema ter um LLM)_.
- Um utilitário de Make (`apt install make` no Ubuntu, ou WSL/MSYS2 no Windows).

### Como rodar localmente

1. **Instale as dependências com `uv` (Substituto moderno do `pip`)**:
   O projeto utiliza o [uv (da Astral)](https://docs.astral.sh/uv/) por ser escrito em Rust e ser até 100x mais rápido que o `pip` tradicional na resolução de pacotes Python.

   Se não tiver instalado, instale via `pip install uv`. Em seguida, rode:

   ```bash
   make dev
   ```

2. **(Opcional, porém Útil) Treinar os Modelos localmente**:
   Verifique se a pasta de modelo está com o `best_model.pkl`. Você pode rodar um pipeline novo do zero que vai re-estruturar hiperparâmetros Random Forest/XGBoost.

   ```bash
   make train
   ```

3. **Suba os Serviços**:
   Você precisará de três terminais paralelos rodando na raiz:
   - **Terminal 1 (Backend API)**:
     ```bash
     make api
     ```
   - **Terminal 2 (Model Tracker MLFLow - Opcional se for não ver estatísticas)**
     ```bash
     mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
     ```
   - **Terminal 3 (UI do Cliente Final)**
     ```bash
     make app
     ```

---

## 4. Como Navegar e Testar a Interface 🧪

Com o projeto ativo (seja de modo Docker ou Local), tudo roda amarrado.
Acesse o navegador nas seguintes rotas:

### 🌟 Visão do Cliente e Análise / Painel Interativo

Acesse: **[http://localhost:8501](http://localhost:8501)**

- O **Streamlit** se abrirá. Você pode brincar na aba lateral esquerda definindo:
  - Limite financeiro (R$/NT$).
  - Idade, casamentos, e meses do histórico.
  - Pressione \`Analisar Risco\`.
  - **O que validar no teste?**
    - Veja o painel mudando seu Gradiente `Aprovado` (Verde) e `Negado` (Rubro).
    - Atente-se à tabela **(Top Fatores Determinantes SHAP)** — Observe qual variável determinou mais o risco algorítmico do modelo treinado.
    - Aguarde mais 4 a 10 segundos, no bloco debaixo uma **Explicação em Linguagem Natural** irá fluir gerada via IA, dissecando e "ensinando" o cliente financeiro a como se endividar menos e expondo do pq recusaram ou deram crédito.

### 🔌 Teste dos Endpoints da Interface de Programação

Acesse: **[http://localhost:8000/docs](http://localhost:8000/docs)**

- É a UI automática do **FastAPI (Swagger)**.
- Abra o endpoint `POST /explain` e clique em `Try it out`.
- Role e aperte em "Execute". Isso permite que você simule como uma aplicação consumiria as decisões do modelo cruamente, mostrando o JSON das aprovações.
- Teste também ver o `GET /health` do sistema, ele acusará `{ "status": "ok", "version": "1.0.0" }`.

### 🗃️ Checando do Motor de Treinamento

Acesse: **[http://localhost:5000](http://localhost:5000)** (Disponível sempre via Docker ou caso tenha processado local)

- Se você acessar por este portal, a visão do MLFLow revelerá todos os experimentos e simulações executados pela base, rastreando `ROC AUC` e `F1 Score` em logs interativos.

---

## 🛑 Limpando a Casa e Encerrando

Para derrubar da memória e parar, utilize:

```bash
# Caso tenha rodado o Docker Opção A
make docker-down
# Ou
docker-compose down

# Para realizar limpeza rigorosa de cache/variáveis da linguagem
make clean
```
