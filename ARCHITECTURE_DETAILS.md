# 🏗️ Detalhamento Técnico: Credit Guard

Este documento fornece uma visão profunda das bibliotecas utilizadas no projeto **Credit Guard** e explica como cada uma contribui para o funcionamento total do sistema.

---

## 📚 Bibliotecas e Suas Funções

### 1. Pandas (`pandas`)

- **O que faz:** É a biblioteca padrão para manipulação e análise de dados tabulares em Python.
- **Como é usada no projeto:**
  - **Carga de Dados:** Carrega o dataset UCI de cartões de crédito.
  - **EDA (Análise Exploratória):** Usada nos notebooks para identificar correlações, valores faltantes e distribuições.
  - **Engenharia de Features:** Criação de novas variáveis, tratamento de dados categóricos e normalização.
  - **Interface API/Frontend:** Conversão de inputs do usuário em DataFrames para a predição.

### 2. Scikit-learn (`scikit-learn`)

- **O que faz:** Ferramenta essencial para mineração e análise de dados, oferecendo algoritmos de classificação, regressão e ferramentas de pré-processamento.
- **Como é usada no projeto:**
  - **Pipeline de Treino:** Divisão dos dados em treino e teste (`train_test_split`).
  - **Métricas:** Cálculo de performance (`ROC-AUC`, `F1-score`, `Accuracy`).
  - **Calibração:** Uso do `CalibratedClassifierCV` para garantir que as probabilidades retornadas (ex: 0.85) correspondam à probabilidade real de inadimplência.
  - **Preprocessing:** Escalonamento de variáveis e codificação de labels.

### 3. XGBoost (`xgboost`)

- **O que faz:** Uma implementação otimizada de _Gradient Boosting_ projetada para ser altamente eficiente, flexível e portátil.
- **Como é usada no projeto:**
  - É um dos **modelos principais** de predição. Ele é excelente para capturar relações não-lineares complexas nos dados financeiros que modelos mais simples (como Regressão Logística) poderiam ignorar.

### 4. LightGBM (`lightgbm`)

- **O que faz:** Outra estrutura de _Gradient Boosting_ que usa algoritmos baseados em árvore, mas com crescimento por folha (_leaf-wise_), sendo frequentemente mais rápido que o XGBoost em grandes volumes de dados.
- **Como é usada no projeto:**
  - Usado para **comparação de modelos**. No Credit Guard, o LightGBM é frequentemente o modelo com melhor performance em termos de velocidade e métricas de erro.

### 5. Optuna (`optuna`)

- **O que faz:** Uma estrutura de otimização de hiperparâmetros automática, que utiliza busca Bayesiana para encontrar as melhores configurações para os modelos.
- **Como é usada no projeto:**
  - **Sintonia Fina:** Em vez de chutar valores para o modelo (como profundidade da árvore ou taxa de aprendizado), o Optuna testa dezenas de combinações de forma inteligente para maximizar o `ROC-AUC`.

### 6. Imbalanced-learn (`imbalanced-learn` / SMOTE)

- **O que faz:** Biblioteca projetada para lidar com datasets onde uma classe (ex: "bons pagadores") é muito maior que a outra ("inadimplentes").
- **Como é usada no projeto:**
  - **SMOTE:** Gera dados sintéticos da classe minoritária (inadimplentes) durante o treino para que o modelo não fique "viciado" em dizer que todo mundo é bom pagador só porque eles são a maioria no dataset.

---

## ⚙️ Funcionamento Total do Projeto

O **Credit Guard** opera como um ecossistema integrado:

1.  **Fase de Treinamento (Offline):**
    - O script `train.py` (ou notebook) carrega os dados com **Pandas**.
    - O **SMOTE** balanceia os dados.
    - O **Optuna** executa dezenas de rodadas de treino comparando **XGBoost** e **LightGBM**.
    - O melhor modelo é salvo na pasta `/models` junto com os metadados do **MLflow**.

2.  **Fase de Predição (Online):**
    - A **API FastAPI** carrega o modelo salvo.
    - Quando um novo cliente é enviado (via JSON), a API limpa os dados e pede uma predição.
    - O modelo calibrado retorna a probabilidade de risco.
    - O **SHAP** entra em cena para calcular a importância de cada atributo para aquela decisão específica.

3.  **Fase de GenAI (Explicabilidade):**
    - Os valores do SHAP (técnicos e numéricos) são enviados para um **LLM** (Ollama ou Groq).
    - O LLM traduz: _"Este cliente tem 80% de risco porque seu saldo devedor nos últimos 2 meses cresceu rápido demais"_.

4.  **Interface e Auditoria:**
    - O usuário interage pelo **Streamlit**.
    - Cada decisão é salva em uma trilha de **Auditoria** para que, se alguém questionar por que o crédito foi negado, o sistema tenha o registro completo do motivo técnico e da explicação humana dada na época.
