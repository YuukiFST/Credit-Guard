"""
Router de predição de risco de crédito.

Endpoint principal: POST /predict
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.config import settings
from src.data.schemas import CreditClientInput, PredictionOutput
from src.features.engineer import FeatureEngineer
from src.audit.logger import AuditLogger

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
    from src.models.trainer import load_model
    from src.explainability.shap_explainer import SHAPExplainer

    request_id = str(uuid.uuid4())
    used_threshold = threshold or settings.model_default_threshold
    logger.info(f"Predição solicitada | request_id={request_id}")

    try:
        model = load_model()
        fe = FeatureEngineer()

        df = client_data.to_dataframe()
        df_engineered = fe.transform(df)
        proba = float(model.predict_proba(df_engineered)[:, 1][0])  # type: ignore[union-attr]
        decision = "NEGADO" if proba >= used_threshold else "APROVADO"

        explainer = SHAPExplainer(model)
        shap_values = explainer.explain_instance(df_engineered)
        top_factors = explainer.get_top_factors(
            shap_values, feature_names=list(df_engineered.columns), n=5,
        )

        # Audit trail
        audit = AuditLogger()
        audit.log_decision(
            request_id=request_id,
            decision=decision,
            probability=proba,
            threshold=used_threshold,
            top_factors=top_factors,
        )

        return PredictionOutput(
            client_id=request_id,
            default_probability=round(proba, 4),
            decision=decision,
            threshold_used=used_threshold,
            top_factors=top_factors,
            model_version="1.0.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Erro na predição | request_id={request_id} error={e!s}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno na predição: {e!s}",
        ) from e
