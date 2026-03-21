# 🎓 Documentação Técnico-Educacional: Credit Guard

Este documento é uma análise técnica profunda do projeto **Credit Guard**. Ele foi escrito para estudantes do 1º semestre de Ciência da Computação que já possuem familiaridade com lógica de programação (variáveis, funções, estruturas de controle), mas que buscam compreender a engenharia por trás de um sistema de software em nível de produção.

A regra fundamental desta documentação é a precisão: todo conceito arquitetural ou ferramenta será formalmente definido em sua primeira ocorrência, compreendendo o "como" e o "porquê" de seu funcionamento computacional real.

---

## 📌 Visão Geral do Projeto

### O Problema Tecnológico e de Negócio

O projeto resolve o problema de modelagem de **risco de crédito**, que consiste em prever matematicamente a probabilidade de um mutuário não pagar sua fatura de cartão de crédito (um evento chamado de _default_ ou inadimplência).

Para resolver isso, o sistema orquestra:

1. A inferência multivariada em algoritmos de Machine Learning (Modelos baseados em Árvores de Decisão).
2. O cálculo da explicabilidade matemática de cada predição.
3. A integração de I/O de rede (processamento Assíncrono) para Grandes Modelos de Linguagem (LLMs).

### Arquitetura Geral

O sistema adota uma arquitetura **Cliente-Servidor (Client-Server)** distribuída. Uma **arquitetura** define como os componentes do software são divididos e como interagem. Na arquitetura Cliente-Servidor, um programa (o Cliente) faz requisições de recursos ou cálculos, e outro programa (o Servidor) as processa e devolve os resultados.

Neste projeto, o Cliente é o aplicativo _Streamlit_, que captura os dados digitados na tela. O Servidor é a _FastAPI_, que contém toda a lógica de negócio e modelos matemáticos. Eles não se misturam no mesmo código; rodam independentes.

### Diagrama de Fluxo de Dados e Fronteiras lógicas

```text
+----------------------+                       +----------------------------------+
|   Frontend Client    |                       |      Backend (API REST)        |
|     (Streamlit)      |                       |         (FastAPI)            |
+----------+-----------+                       +-------+--------------------------+
           |                                           |
           | HTTP POST: /predict                       |
           | Payload: JSON                             |
           v                                           v
+----------------------+                       +---------------+
| Serialização  (Dict) |---------------------->| Roteador de   | (Validação de tipo strict via Pydantic)
| -> Byte Stream JSON  | <---------------------| Previsão      |
+----------------------+        HTTP 200       +-------+-------+
                                Resposta JSON          |
                                                       v
                                               +---------------------------------+
                                               | Modulo de Feature Engineering   |
                                               | (Transformação de Matriz Numpy) |
                                               +-------+-------------------------+
                                                       |
                                               +-------v-------------------------+
                                               | Model Inference (LightGBM)      |
                                               | (Retorna Float. Ex: 0.85 risco) |
                                               +-------+-------------------------+
                                                       |
                                               +-------v--------------------------+
                                               | SHAP Explainer (Subrotina C++)   |
                                               | (Retorna Array de contribuições) |
                                               +-------+--------------------------+
                                                       |
                                               +-------v--------------------------+
                                               | LLM Integration (I/O HTTP)       |
                                               | (Conexão Assíncrona via Sockets) |
                                               +----------------------------------+
```

**Termos utilizados no diagrama:**

- **I/O (Input/Output):** Operação de entrada e saída. Refere-se a qualquer comunicação que sai do processador principal e vai para a memória, disco ou placa de rede. Ler um arquivo ou enviar dados pela internet são operações de I/O e geralmente são ordens de grandeza mais lentas que cálculos matemáticos (CPU-Bound).
- **HTTP (Hypertext Transfer Protocol):** Protocolo base da internet. Define um conjunto de regras para comunicação em rede. O método `POST` do HTTP sinaliza que o cliente deseja enviar novos dados no corpo da mensagem ao servidor. O código de sucesso `200` sinaliza que o servidor processou tudo sem erros.
- **REST (Representational State Transfer):** Um estilo de arquitetura de software para sistemas distribuídos. Uma **API REST** padroniza a comunicação HTTP para que sistemas conversem usando verbos (GET, POST, PUT, DELETE) e _Endpoints_ (URLs finais, como `/predict`), e quase sempre trafeguem textos em formato estruturado explícito.
- **JSON (JavaScript Object Notation):** Um formato de arquivo de texto estruturado utilizando pares de chave-valor. O Frontend (Streamlit) envia números estruturados em JSON, como `{"age": 25, "limit": 1000.0}`, que viajam em texto e são lidos do outro lado do processo.
- **API (Application Programming Interface):** Um contrato em código que determina as assinaturas de funções ou pontes de rede onde um software se permite ser acessado ou invocado por outro software ("Eu aceito idade em inteiro, e em troca prometo retornar um risco em decimal").

---

## 🗂️ Estrutura de Pastas e Arquivos

O projeto utiliza um padrão de organização de diretórios rígido visando minimizar o acoplamento. **Acoplamento** é o grau de dependência entre módulos diferentes; se alterar o sistema de logs quebra o salvamento no banco de dados, o acoplamento está alto.

- 📁 `app/`: Isola o código de front-end, onde reside o `streamlit_app.py`. A decisão arquitetural de separar do `src/` garante que as dependências relativas a Interface Visão do Usuário (UI) não se misturem à lógica do servidor. Se `app/` fosse fundido com `src/`, atualizar uma biblioteca de gráficos poderia inadvertidamente conflitar as bibliotecas do servidor core, violando o isolamento de ambientes.

- 📁 `src/`: É a raiz controlada da Lógica Servidora, dividida intencionalmente por camadas de domínio (separação lógica das obrigações).
  - 📁 `api/`: Camada de borda da rede. Todo arquivo aqui (ex: `routers/predict.py`) deve lidar exclusivamente com recebimento de tráfego HTTP, extração de parâmetros da URL e devolução de códigos (HTTP 500, etc). Nunca há cálculo estatístico aqui.
  - 📁 `models/`: Módulo (`trainer.py`, `evaluator.py`) recluso encarregado exclusivamente pela inicialização estrutural e matemática profunda do _Machine Learning_, mantendo os ciclos de treinamento afastados das rotas da API.
  - 📁 `data/`: Contém os Contratos de Dados (`schemas.py`) e sub-rotinas de ingestão (`loader.py`) em disco, operando a fronteira tipificadora do sistema.
  - 📁 `features/`: Isolamento matemático com a classe `engineer.py`, responsável pela transformação vetorial (matrizes) antes dos algoritmos preditivos receberem inputs estatísticos.

- 📄 `config.yaml` / `.env`: Arquivos voltados a isolar configurações variáveis da Aplicação.
- 📄 `Makefile`: Arquivo de build de diretrizes herdado nativamente por processamento _Make_ do ecossistema POSIX, abstraindo múltiplos comandos CLI (_Command Line Interface_).
- 📄 `Dockerfile` / `docker-compose.yml`: Arquivos-texto instrumentando como virtualizar o Kernel para o processo inteiro via container.

---

## 🧰 Tecnologias e Ferramentas

### 1. Python como Linguagem Host

- **Conceito estrutural**: Antes da ferramenta, precisamos de um ambiente de execução que possa realizar intermediação (abstrair alocadores em baixo nível num compilador/interpretador) entre os desenvolvedores modelando a matemática pesada e os chips litográficos lidando com bits.
- **O que é e como funciona**: Python é uma linguagem multiplataforma de altíssimo nível, interpretada em tempo de execução via máquina virtual (instruções transacionadas via bytecodes ao processo CPython). Seu modelo de execução possui _tipagem forte_ e alocação via _Garbage Collector_ com contagem referencial.
- **Por que foi escolhida (Comparação)**: Em backend puro de rede, Golang ou Rust trariam vantagens cruciais de _multithreading_ veloz não bloqueadas pelo GIL (Global Interpreter Lock, regra do Python que limita o bytecompilador a uma _thread_ nativa do SO por vez). A escolha se justifica estritamente porque o ecossistema maduro em ML numérico (`numpy`, manipulações na biblioteca libomp C++) consolida C/C++ envelopados transparentemente no uso do Python, sinergizando as equipes Backend CPython com Data Science.

### 2. FastAPI (Web Framework)

- **O que é / Como funciona internamente:** É um _Framework_ Web. Um Framework dita as regras arquiteturais, provê as funções da sintaxe para recebimento na porta TCP e padroniza injeção. O FastAPI implementa o protocolo ASGI (_Asynchronous Server Gateway Interface_). Diferente dos modelos puramente procedurais, ele roda sobre um _Event Loop_ em baixo nível — ele não congela aguardando sub-retornos demorados, mas transita o fluxo da pilha.
- **Comparação (FastAPI vs Flask vs Django)**:
  - **Django** impõe M.V.C robusto (Modelo/Visão/Controle), excessivo para endpoints soltos; traz ORM de tabela não relacional que este projeto não necessita.
  - **Flask** repousa tipicamente em WSGI (Síncrono/Bloqueante). A requisição HTTP tranca a _worker thread_ num lock de kernel quando uma rota Flask invoca a requisição de Rede remota Groq de processamento (que demora 2 segundos para retornar).
  - A adoção do FastAPI lida magistralmente com concorrência _network/I/O-Bound_ e embute interpretação dinâmica via anotações, viabilizando o projeto de forma superior em performance bruta.

### 3. Pydantic (Validador e Tipagem Estrita em Runtime)

- **Qual problema resolve:** Linguagens estruturadas estaticamente como C++ ou Java embutem validações fortes de tipo (_Type Safety_) em compilação, o que aborta construção se passarmos um _String_ num Inteiro `int`. Como Python deduz dinamicamente os tipos de variáveis no código de acordo com o que guardam (_Duck Typing_), sem controle o programa inteiro cederia na etapa das matrizes quando um atacante enviasse pela URL as idades preenchidas como `idade="Quarenta e Dois"`. Pydantic impõe barreira matemática preventiva em tempo de execução (_Runtime_).
- **Como é Usado Neste Projeto:** Localizado no módulo contíguo abstrato (`src/data/schemas.py`).

  ```python
  from pydantic import BaseModel, confloat

  class PredictionOutput(BaseModel):
      # Type Hint puro na linguagem avisaria à IDE a intenção,
      # Pydantic FORÇA ativamente a variável default_probability: cast para o float
      # E caso o valor numérico quebre ele devolve pânico sem derrubar a API.
      default_probability: float
      decision: str
  ```

### 4. Containerização e Docker

- **Conceito técnico de base:** Problema do isolamento do ambiente de dependência. Ao desenvolvermos o app na interface do macOS instalando as libs `xgboost.dll` e `.so` correspondentes e tentarmos rodar no servidor em produção do Azure rodando Linux Ubuntu, o sistema carece das bibliotecas compartilhadas padrão (glibc), acarretando fatalidade instantânea em _Shared Object missing_.
- **Como funciona:** Docker NÃO é uma Máquina Virtual (VM) emulando hardware completo com núcleo de sistema via hypervisor. Docker orquestra instruções avançadas do Unix/Linux kernel. Ele encapsula processos em um **Container** isolado via _Namespaces_ (falsificando pro código Python de que os únicos arquivos abertos nos diretórios ou redes são os contidos ali restolhados em chroot virtual) e restringe uso implícito via _cgroups_ (Control Groups de memória/Processador).
- **Docker Compose:** Enquanto as Imagens (receitas engessadas contendo libs/OS/kernel base) Docker lidam um-contra-um, o Compose orquestra Subredes Isoladas virtuais para que Múltiplos processos de Docker possam dar _Ping_ uns nos outros.

### 5. Gerenciamento de Dependências: A Revolução do `uv`

- **O Problema do `pip` Clássico:** Projetos Python tradicionais utilizam o `pip` para baixar e instalar bibliotecas (como `pandas` ou `fastapi`). O problema é que o `pip` é escrito em Python puro, o que torna a resolução de dependências (verificar se a versão X da biblioteca A é compatível com a biblioteca B) um processo lento, bloqueante e obsoleto para CI/CDs modernos.
- **Como o `uv` (da Astral) resolve:** Este projeto adota o `uv`, um instalador moderno escrito em **Rust** (linguagem de programação de sistemas que compila direto para código de máquina). O `uv` é até 100x mais rápido que o `pip` padrão, possuindo um cache global de sistema inteligente.
- **Lição de Arquitetura Sênior:** Desenvolvedores seniores não se prendem apenas ao código do servidor; eles otimizam a _Esteira de Deploy_. Substituir o `pip` pelo `uv` no `Dockerfile` e no `Makefile` deste projeto reduziu o tempo que o container demora para ser montado de minutos para segundos, economizando processamento computacional em larga escala.

---

## 🔄 Fluxo de Execução e Funcionalidades do Projeto

Seguiremos o _Tracing_ sequencial de uma requisição trafegada desde o acionamento visual até as transformações estruturais de processadores paralelos:

1. **Serialização e Acoplamento da Requisição `[APP / STREAMLIT]`**
   O Streamlit atua captando inputs UI (Variáveis). Utilizando a lib interna instanciadora _httpx_, invoca em I/O bound pelo IPv4 na conexão porta (`8000`) efetuando postagem JSON.

2. **Interceptação de Endpoint e Decorator Engine `[API / FASTAPI]`**
   **Arquivo:** `src/api/routers/predict.py`.
   - **O Decorator (`@router.post`)** em Python embrulha uma função (_closure_) e modifica o status subjacente da função, definindo semânticas extra sem alterar a implementação de `predict_credit_risk`. Ele matricula internamente a função à escuta de rotas no FastAPI.

   ```python
   @router.post(
       "/predict",
       response_model=PredictionOutput, # Contrato de Saida - Pydantic garante restrição
   )
   async def predict_credit_risk( # Declarada Assíncrona. Ponto em que o Event Loop delega e coordena
       client_data: CreditClientInput, # Validação Pydantic injetada para as entradas
   ) -> PredictionOutput:
   ```

3. **Injeção Lógica nas Árvores Preditivas e SHAP Explainer `[FEATURES -> MODELS]`**
   **Arquivo:** `src/api/routers/predict.py`.
   A API aciona métodos iterando os algoritmos carregados estatísticos via Numpy subjacente de Matriz com _Feature Engineer_. Transformações do Feature process (`fe.transform`) mutacionam dimensionalmente matrizes bidimensionais em tempo ($O(n)$) e delegam inferência probabilística (_forward pass_) aos vetores processuais XGBoost em árvore do Backend via `model.predict_proba`.

   ```python
       model = load_model()
       fe = FeatureEngineer()

       # A deserialização estrita no Pandas encapsula as listas nativas para alocas
       # C-contiguous em baixo nível pra sub-chamada na inferência Numpy/C++ do LightGBM.
       df = client_data.to_dataframe()
       df_engineered = fe.transform(df)
       # [:, 1][0] Seleciona slice bidimensional sub-array correspondente (A Aresta de erro / default 1)
       proba = float(model.predict_proba(df_engineered)[:, 1][0])
   ```

4. **I/O Bound de Explicações LLM e Assincronismo `[LLM / MODELOS: PHI-3 & LLAMA-3.1]`**
   **Arquivo:** `src/llm/client.py`. Após capturar as chaves métricas que causaram o "Risk Default", o sistema repassa descrições asíncronamente pra processamento neural. O projeto utiliza o modelo **Phi-3 Mini** (um SLM - Small Language Model da Microsoft) via Ollama localmente, e o **Llama 3.1 8B** da Meta como fallback via Groq. No trecho inferior contido na classe geradora a função _await_ entrega aos Sockets O.S sob gerência HTTPX:
   ```python
       async def generate(self, prompt: str, system_prompt: str = "") -> str:
           # Contexto que fecha a thread network ao final das iterações
           async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
               # Ponto crítico ASGI: a função inteira PAUSA aqui, desbloqueando cpu inteira para OS.
               # Aguarda a recepção completa pacotes socket/JSON enviados iterativamente do backend API da rede.
               response = await client.post(...)
               response.raise_for_status()
               return response.json()["choices"][0]["message"]["content"]
   ```

---

## 🤖 Machine Learning e IA: Aprofundamento Técnico Restrito

A seção preditiva adota a variante não-paramétrica algorítmica supervisionada enquadrada sob indução matemática.

### Modelos de Referência: O Ciclo Supervisionado Funcional

Neste projeto utilizamos e comparamos Árvores de Classificação baseados em conjuntos (**Ensemble**), operando inferência em _Machine Learning_ **Supervisionado**. No paradigma Supervisionado mapeamos um conjunto contendo Variáveis Independentes (**Features** – como idade e renda) para Variáveis Dependentes/Alvos em tempo histórico (**Labels** – o _Simulação/Status Pago ou Prejuízo Default_ na base inicial de meses atrás).

- **Treinamento:** Processo em que o modelo ajusta iterativamente os parâmetros algébricos da matriz minimizando o erro residual iterativo medido em função heurística (Loss Function).
- **Inferência:** Dado parâmetros estabilizados e serializados arquivados, projeta novos DataFrames nunca antes mapeados na heurística para derivar resultados em tempo constante, retornando o Risco Absoluto e Relativo.
- **Serialização**: Ato de transformar objeto em memória (um classificador estatístico em árvores contendo regras de milhões de IFs enlaçados encadeados baseados no Kernel) numa _String Byte Stream_. Usamos o `Pickle/Joblib` na classe _Trainer_ para converter a rede do Scikit-Learn e arquivar em Disco persistido `best_model.pkl` de modo a abstraí-lo novamente (`load_model`) pro API em subida.

### O Cérebro: Gradient Boosting (XGBoost/LightGBM)

- **Árvore de Decisão (_Decision Tree_):** Forma Grafos direcionados acíclicos com partições nos atributos limitantes, processando nós intercessores separadores iterando divisibilidade heurística métrica de purezas da matriz (_Critério de entropia de Shannon_ ou _Impureza Gini_). Ao subdividir, cada ramificação do nodo busca isolar a homogeneidade nos atributos remanescentes.
- **Ensemble Learning:** Empregar uma única Arvore de decisão suscita alta taxa de sobreajuste de treino estatístico (_Variância Alta/Overfitting_ – decora o dataset ao invés de descobrir padrão limitador de generalizações na matemática) versus (_Bias/Viés alto_ em arvores ralas – incapazes de aprender). Ensemble mescla e processa em pipeline instâncias independentemente modelando _Arvores Fracas_ subdimensionadas, calculando e aglomerando vetores resultantes estatisticamente.
- **Gradient Boosting:** Se contrapõem às formatações de Bagging (como as Random Forest onde as árvores decidem independentes) treinando árvores sequenciais baseadas nas divergências da Anterior. A Função da Perda Logarítmica avalia a desconexão paramétrica do pretexto da árvore Base contra a matemática real do Label do Banco (`y - yHat`). Aplicando Derivativo matemático e descendências graduais (O Gradiente), a próxima Sequência SubArbórea do pipeline treinará exclusivamente prevendo os pseudo-residual gradientes residuais decaídos. Modelação extrema com correção ativa!
- **XGBoost e LightGBM – Divergências Técnicas:**
  O XGBoost clássico cresce todos as ramificações de mesmo estagio simultaneamente em amplitude _Level-wise Growth_.
  A adoção secundária do framework **LightGBM**, em subcamada algorítmica de expansibilidade, propõe expansões folha-a-folha assimétricas (**Leaf-Wise Growth**) localizados diretamente ao vértice de maiores decaídas graduacionais na Loss Function, sacrificando homogeneidade para escalar as predições iterativas numa assíntota veloz. Acrescenta-se a este algorítmo reduções massimais ao O(n) pelo seu EFB (_Exclusive Feature Bundling_ aglutinando colunas de dados nulas para redução de escaneio paramétrico e o recorte estrito via GOSS (_Gradient-based One-Side Sampling_) em que descartam amostras computacionais matriciais lineares já resolvidas do kernel e operam subamostragem direcionadas apenas à porção ineficiente dos dados.

### Explicabilidade em Processos de Inferência (SHAP)

Quando um algoritmo ensemble aprova a instância não linear com predição limítrofe, requer-se inferência interpretável, um gargalo algorítmico do modelo "Caixa-Preta" restrito. Utilizamos cálculos agregadores do módulo SHAP.

- **Valores de Shapley:** Radicados pelo matemático americano da Teoria de Jogos, definem computacionalmente matrizes retributivas sobre retornos ou prêmios combinatórios agregados (Ex, "Quem cooperou marginalmente para chegar no risco de 0.85").
- **SHAP em Machine Learning:** Na abordagem por SHAP os recursos subjacentes as Features de crédito são transpostas no _Jogador Matemático Cooperativo_. Subtraímos as contribuições calculando permutações exponenciais contra uma base media algorítmica expectativo no núcleo zero $E(f(z))$. Se Feature de _Mês em Atraso 02_ reporta _Shap_Value positivo (+.20)_ este atua incrementando as probabilidades rumo a Classificação 1 (Default Positivo) descredibilizando o ator. A implementação contígua C++ injeta _TreeExplainer_ local com formula direta para as equações baseadas em percursos polinomiais para contornar custo exponenciação de complexidade teórica NP-Hard original de formula em grafos ramificados de arvore percorrida.

### Interpretações Generativas LLM: Pós-Processamento Natural

- **O Large Language Model (Transformer Architecture):** Componente neural submetido baseando na revolução da camada atencional (_Attention Is All You Need_ da arquitetura _Self-Attention_). Redes calculam densamente através de blocos relacionamentos inter e posições de Tokens (Partições estatisticas sub-palavras fracionadas das _Strings_ numéricas transacionadas nas requisições semânticas da array numérico $P(w|w_{-n:t})$ iterativo ao processo auto-regressivo de inferência para adivinhar por vetores multi dimensionais a proxima estrutura léxica). Todo o contexto das variações provindas do SHAP transpostas no Payload injetado compõem o contexto injetável ou **Prompt** matricial delimitador contextual instruído ao bloco Decoder generativo adjacente.
- **A Decisão do Orquestrador de Modelo Concorrente em API:**
  Inferir no modelo **Phi-3 Mini** internamente locado ao processador base (_Self Hosted_) no daemon da lib _Ollama_ detém restrições da capacidade VRAM/RAM (Acesso da Memória unificada dependente em hardwares consumer), porém suprime em escala as Latências (RTT em redes locais são sub-microsegundos de overhead na _system call/sockets unix_) isentando também custos mensais corporativos limitados pelo Hardware Cap rate-limiting natural ou exposição da camada de Segurança aos Dados corporativos sensíveis externados.
  Caso se deflagre limitação local/CPU travada com esvaziamento computacional, orquestramos em Circuit-Breaker o Acoplamento externo na malha Groq utilizando o modelo **Llama 3.1 8B** (Inferência alocada subjacente a LPU's com velocidade extrema na reposta sequencial gerativa transacional em Tokens-per-Second). Desvantagens incluem restrições I/O externas via conexções seguras em SSL (Overhead de Handshake em latência subjacentes dependentes dos nós das operadoras BGP) e acoplamento a precificações por milhares de uso estrito via keys criptografadas expostas na requisição subjacente.

---

## ⚙️ Arquitetura de Software: Racionalizações Críticas

Por trânsferências intrincadas em padrões corporativos:

- **Separação Frontend/Backend (API vs View):** Acoplamento remete do escopo interligações no mesmo subsistema. Criar instâncias no Python fundindo Front visual das matrizes das árvores quebra escalabilidade de subidas simultaneas no Cluster para CPU. Desacoplamo-os pois se torna factível orquestrar em infra na internet 3 Backend Models Server operacionais de API que processam o gargalo pesado de Machine Learning atendendo ao escalonamento isolado e simultâneo único pro Frontend das visualizações estáticas sub-recurso. Alterar os estilos e cores em front nos relatórios Streamlit nunca impacta os módulos de Regulação Estatísticos do Trainer no código Core.
- **Isolamento de Credenciais em Variáveis de Ambiente:** No módulo principal injetivo de instâncias para conexões de Backend o código exclama por Tokens ou Segredos autorizados do Groq (A Chave de API base `GROQ_API_KEY`). A configuração das alocações da Memória Virtual (_Environment Variables_) no subsistema Operacional invés das strings declarativas "hardcoded" (`api_key = "MINHA_SENHA"`) blinda o código à não-vazamento nas estruturas de repositórios git-push públicos. Segue conformidades estrita pro subdomínio Twelve-Factor Architecture que garantem imutabilidade sistêmica das implentações. Na versão de Docker os orquestradores repassam os Environment Variable por meio da serialização paralela contida em `.env` que transitam para as execuções restritas alocadas _forkadas_ sem vazar pra Base persistente compilada e arquivada via _Docker Commit_.

---

## ⚙️ Compilação das Bases: Entenda como Testar/Rodar as Camadas Isoldas

A aplicação utiliza embutimentos via container para subir camadas na orquestração:
**Pré-Requisito Estrito:** Docker Engine Client & Daemon rodando para interface Cgroup de restrição de ambiente isolado sub.

Execute via terminal de sua distro ou ambiente os scripts de ativação da orquestração virtualizada do Compose referida no arquivo base yaml pre-declarado contendo todo workflow sistêmico dos nós locais alocados.

- `$ docker-compose up -d --build`

**O processamento da Infraestrutura engatilhada local:**

1. Os paramêtros declarativos _build context_ instanciará _sublayers_ (Baixando o Ubuntu e libs Python Base referiadas pela Image no _Dockerfile_)
2. Configura uma **Sub-rede em Bridge Mode isolada**, ou seja cria a interface do roteamento para os IPs das portas abertos referenciados nos modulos internos dos containers
3. Expõe _Port Forwardings_ no Firewall redirecionado as inter-portões para Host Interface de sub uso (`8000:8000`), liberando que o Sistema real em Browser do Windows acesse diretamente a rota isolada escutada do Linux pelo Container em http://127.0.0.1:8000/.
4. Utiliza **Volume Bind Mount** nos diretórios de instâncias locais mapeados aos Containeres da subpasta `modelos` para subescrever referências locais em Host na permanência do disco do container virtual abstraído limitante preservando-os mesmo desligado à execução container em kernel (`Volumes`).

---

## 🐛 Erros Comuns Estruturais e Diagnóstico Sub-camada Técnico

No contexto Backend Rest em Python rodando Sub-containers, podem emergiram falhas não intuitivas e os respectivos _TracesBack_:

**Erro I/O:** `httpx.ConnectError: [Errno 111] Connection refused` no script da tela do Frontend visual

- **Causa Real Base Kernel:** Uma exceção levantada via Socket na biblioteca TCP das bibliotecas socket C por Python após efetuar um pacote em SYN (Synchronization tcp route request protocol network), o socket do _cliente_ Streamlit de Interface atingiu com precisão na Interface loopback a Porta exata designada do Host, todavia e deparou-se que não existe sub-sistema Kernel Listeners (Nenhum Daemon do servidor) que processe em status ativo naquilo, enviando imediatamente TCP RST em resposta o que traduz as falhas _Errno 111_.
- **Diagnostico:** O Container da API Backend caiu com erro não tratado fatalmente, ou nunca finalizou _up start bootloader process_. Utilize as leituras restritivas nas execuções de log (Ex `docker logs <Nome do Container API_backend>`). Reviva ou re-suba via execuções no terminal com `make api` nas malhas da port forwarding se trabalhando sem dockers em desenvolvimento de host raw script na local host nativa pura de Kernel.

**Erro Categoria Ilegível BaseDados/Matriz:** `TypeError: unhashable type: 'dict' no Engineer processing` ou `ValueError in Arrays Pandas dtype shape`

- **Causa Real Matemática da Array:** Na arquiteturação dos Arrays no Numpy das sub funções e rotinas as bibliotecas encapsuladas e os processamentos e serializações (Como JSONs nas APIs de repasse ao pipeline preditivo nas conversão Dataframes pandas instancidas) operam exclusivamente perante instâncias bidimensionadas que exigem tipos primitivos ou homogeneidade nos Arrays limitados no tamanho de Colunas originais ou na imutáveis nas Tipagens dos objetos _Object type_. O Python acionou Pânico interno (Dtypes).
- **Correção da Linha Estrita:** As execuções perpassadas das Features da Pydantic preveriam falhas se bem estruturadas (Verificações Base Type Checker) Todavia analises referidas à Arrays corrompidas, investigue o trace em Stack frame visualizando as atribuições em Dictionários mal formados provindos do Body JSON restritos não lidos pelo Pydantic com erro em serialização.

---

_Fim do documento analítico. Para os calouros: Compreendam os sistemas escaláveis abstraindo as tecnologias pela base estrutural._
