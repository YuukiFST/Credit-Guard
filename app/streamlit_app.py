"""
Interface Streamlit para análise de risco de crédito.

Frontend interativo que conecta ao modelo de ML
e à API de explicação via LLM.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import httpx
import asyncio


# ─── Configuração da página ───
st.set_page_config(
    page_title="💳 Credit Guard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── CSS customizado ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .denied {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───
st.markdown('<p class="main-header">💳 Credit Guard</p>', unsafe_allow_html=True)
st.markdown("**Análise de Risco de Crédito com ML + GenAI** — Sistema de predição de inadimplência com explicabilidade")

st.divider()


# ─── Sidebar: Parâmetros do cliente ───
with st.sidebar:
    st.header("📋 Dados do Cliente")

    limit_balance = st.number_input(
        "Limite de Crédito (NT$)",
        min_value=1000, max_value=1000000, value=50000, step=5000,
    )
    age = st.slider("Idade", 18, 80, 35)
    sex = st.selectbox("Sexo", options=[1, 2], format_func=lambda x: "Masculino" if x == 1 else "Feminino")
    education = st.selectbox(
        "Escolaridade",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Pós-Graduação", 2: "Graduação", 3: "Ensino Médio", 4: "Outro"}[x],
    )
    marriage = st.selectbox(
        "Estado Civil",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Casado(a)", 2: "Solteiro(a)", 3: "Outro"}[x],
    )

    st.subheader("📊 Histórico de Pagamento")
    st.caption("Últimos 6 meses (mais recente → mais antigo)")
    pay_labels = {-2: "Sem consumo", -1: "Pago em dia", 0: "Rotativo", 1: "1 mês atraso", 2: "2 meses atraso"}
    pay_history = []
    for i in range(1, 7):
        val = st.selectbox(
            f"Mês {i}",
            options=[-2, -1, 0, 1, 2],
            index=1,
            format_func=lambda x: pay_labels.get(x, f"{x} meses atraso"),
            key=f"pay_{i}",
        )
        pay_history.append(val)

    st.subheader("💰 Faturas e Pagamentos (NT$)")
    bill_amounts = [st.number_input(f"Fatura mês {i}", value=15000, key=f"bill_{i}") for i in range(1, 7)]
    pay_amounts = [st.number_input(f"Pagamento mês {i}", value=2000, key=f"pamt_{i}") for i in range(1, 7)]

    threshold = st.slider("🎯 Threshold de Decisão", 0.1, 0.9, 0.5, 0.05)

    analyze_btn = st.button("🔍 Analisar Risco", type="primary", use_container_width=True)


# ─── Análise principal ───
if analyze_btn:
    with st.spinner("Analisando risco de crédito..."):
        # Montagem payload
        payload = {
            "limit_balance": float(limit_balance),
            "age": age,
            "sex": sex,
            "education": education,
            "marriage": marriage,
            "pay_history": pay_history,
            "bill_amounts": [float(b) for b in bill_amounts],
            "pay_amounts": [float(p) for p in pay_amounts],
        }

        try:
            # Tenta chamar a API
            response = httpx.post(
                f"http://localhost:8000/predict?threshold={threshold}",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()

                # Métricas principais
                col1, col2, col3 = st.columns(3)
                with col1:
                    decision_class = "approved" if result["decision"] == "APROVADO" else "denied"
                    st.markdown(
                        f'<div class="metric-card {decision_class}">'
                        f'<h2>{result["decision"]}</h2>'
                        f'<p>Decisão de Crédito</p></div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.metric("📊 Probabilidade de Default", f"{result['default_probability']:.1%}")
                with col3:
                    st.metric("🎯 Threshold", f"{result['threshold_used']:.0%}")

                # Fatores SHAP
                st.subheader("📈 Fatores Determinantes (SHAP)")
                factors_df = pd.DataFrame(result["top_factors"])
                fig = px.bar(
                    factors_df,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="shap_value",
                    color_continuous_scale="RdYlGn_r",
                    labels={"shap_value": "Impacto SHAP", "feature": "Feature"},
                )
                fig.update_layout(
                    height=300,
                    yaxis={"categoryorder": "total ascending"},
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Explicação via LLM
                st.subheader("🤖 Explicação em Linguagem Natural")
                try:
                    explain_response = httpx.post(
                        f"http://localhost:8000/explain?threshold={threshold}",
                        json=payload,
                        timeout=60,
                    )
                    if explain_response.status_code == 200:
                        narrative = explain_response.json().get("narrative", "")
                        st.info(narrative)
                    else:
                        st.warning("Explicação via LLM não disponível.")
                except Exception:
                    st.warning("Serviço de explicação não acessível.")

            else:
                st.error(f"Erro na API: {response.status_code} — {response.text}")

        except httpx.ConnectError:
            st.warning(
                "⚠️ API não disponível. Inicie-a com `make api` ou `docker-compose up`.\n\n"
                "A interface funciona de forma independente quando conectada à API."
            )
        except Exception as e:
            st.error(f"Erro inesperado: {e}")


# ─── Footer ───
st.divider()
st.caption("💳 Credit Guard — ML + GenAI para Análise de Risco de Crédito | v1.0.0")
