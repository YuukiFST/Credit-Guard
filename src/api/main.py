"""
Aplicação FastAPI principal.

Ponto de entrada da API REST de risco de crédito.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from loguru import logger

from src.api.routers import predict, explain, health


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifecycle da aplicação: setup e teardown."""
    logger.info("🚀 API inicializando...")
    yield
    logger.info("🛑 API finalizando...")


app = FastAPI(
    title="Credit Guard API",
    description=(
        "API de predição de risco de crédito com ML + GenAI. "
        "Combina XGBoost com SHAP para explicabilidade e LLM para narrativas."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Registra routers
app.include_router(predict.router, tags=["Predição"])
app.include_router(explain.router, tags=["Explicação"])
app.include_router(health.router, tags=["Saúde"])
