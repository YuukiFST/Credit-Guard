"""
Router de predição de risco de crédito.

Endpoint principal: POST /predict
"""

import uuid
from datetime import UTC, datetime
from typing import cast

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.audit.logger import AuditLogger
from src.config import settings
from src.data.schemas import CreditClientInput, PredictionOutput
from src.features.engineer import FeatureEngineer, align_engineered_to_model

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predição de risco de crédito",
    description=(
        "Recebe dados do cliente, calcula probabilidade de inadimplência, "
        "e retorna decisão com fatores explicativos via SHAP."
    ),
)
async def predict_credit_risk(
    client_data: CreditClientInput,
    threshold: float | None = None,
) -> PredictionOutput:
    """Endpoint de predição de risco de crédito."""
    # Imports lazy — evita carregar MLflow/sklearn ao importar o módulo
    from src.explainability.shap_explainer import SHAPExplainer
    from src.models.trainer import load_model

    request_id = str(uuid.uuid4())
    used_threshold = threshold or settings.model_default_threshold
    logger.info(f"Predição solicitada | request_id={request_id}")

    try:
        model = load_model()
        fe = FeatureEngineer()

        df = client_data.to_dataframe()
        df_engineered = align_engineered_to_model(model, fe.transform(df))
        proba = float(model.predict_proba(df_engineered)[:, 1][0])  # type: ignore[attr-defined]
        decision = "NEGADO" if proba >= used_threshold else "APROVADO"

        explainer = SHAPExplainer(model)
        shap_values = explainer.explain_instance(df_engineered)
        top_factors = explainer.get_top_factors(
            shap_values,
            feature_names=list(df_engineered.columns),
            n=5,
        )
        factors_payload = cast(list[dict[str, str | float]], top_factors)

        # Audit trail
        audit = AuditLogger()
        audit.log_decision(
            request_id=request_id,
            decision=decision,
            probability=proba,
            threshold=used_threshold,
            top_factors=factors_payload,
        )

        return PredictionOutput(
            client_id=request_id,
            default_probability=round(proba, 4),
            decision=decision,
            threshold_used=used_threshold,
            top_factors=factors_payload,
            model_version="1.0.0",
            timestamp=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Erro na predição | request_id={request_id} error={e!s}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno na predição: {e!s}",
        ) from e
