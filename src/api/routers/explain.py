"""
Router de explicação narrativa de crédito.

Endpoint: POST /explain
Combina SHAP com LLM para gerar explicação em linguagem natural.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.config import settings
from src.data.schemas import CreditClientInput
from src.features.engineer import FeatureEngineer
from src.llm.guardrails import safe_generate

router = APIRouter()


@router.post(
    "/explain",
    summary="Explicação narrativa de decisão de crédito",
    description=(
        "Gera uma explicação em linguagem natural para a decisão de crédito "
        "de um cliente, combinando SHAP com narrativa de LLM."
    ),
)
async def explain_decision(
    client_data: CreditClientInput,
    threshold: float | None = None,
) -> dict:
    """Endpoint de explicação via LLM."""
    # Imports lazy — evita carregar MLflow/sklearn ao importar o módulo
    from src.models.trainer import load_model
    from src.explainability.shap_explainer import SHAPExplainer
    from src.llm.client import LLMClientWithFallback

    request_id = str(uuid.uuid4())
    used_threshold = threshold or settings.model_default_threshold
    logger.info(f"Explicação solicitada | request_id={request_id}")

    try:
        model = load_model()
        fe = FeatureEngineer()

        df = client_data.to_dataframe()
        df_engineered = fe.transform(df)
        proba = float(model.predict_proba(df_engineered)[:, 1][0])  # type: ignore[union-attr]
        decision = "NEGADO" if proba >= used_threshold else "APROVADO"

        # SHAP
        explainer = SHAPExplainer(model)
        shap_values = explainer.explain_instance(df_engineered)
        top_factors = explainer.get_top_factors(
            shap_values, feature_names=list(df_engineered.columns), n=5,
        )

        # LLM com fallback
        llm = LLMClientWithFallback()
        raw_narrative = await llm.generate(
            prompt=_build_explain_prompt(decision, proba, top_factors),
            system_prompt=(
                "Você é um analista de crédito experiente. "
                "Explique decisões de forma clara para não-técnicos."
            ),
        )

        # Guardrails
        factor_names = [f["feature"] for f in top_factors]
        narrative = safe_generate(raw_narrative, factor_names, decision)

        return {
            "client_id": request_id,
            "decision": decision,
            "probability": round(proba, 4),
            "threshold_used": used_threshold,
            "narrative": narrative,
            "top_factors": top_factors,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Erro na explicação | request_id={request_id} error={e!s}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na explicação: {e!s}",
        ) from e


def _build_explain_prompt(
    decision: str,
    probability: float,
    top_factors: list[dict],
) -> str:
    """Constrói o prompt para geração de narrativa."""
    factors_text = "\n".join([
        f"- {f['feature']}: SHAP value = {f['shap_value']:.4f}"
        for f in top_factors
    ])
    return f"""
DECISÃO DE CRÉDITO: {decision}
Probabilidade de inadimplência: {probability:.1%}

FATORES DETERMINANTES (ordenados por impacto):
{factors_text}

Gere uma explicação em português que:
1. Explique a decisão de forma clara e não técnica
2. Descreva o papel de cada fator na decisão
3. Oriente o cliente sobre como melhorar seu perfil (se negado)
"""
