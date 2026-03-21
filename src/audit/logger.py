"""
Log imutável de decisões de crédito.

Registra todas as predições para:
- Auditoria regulatória (LGPD Art. 20)
- Debugging e análise post-mortem
- Rastreabilidade de decisões automatizadas
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config import settings


class AuditLogger:
    """
    Logger de auditoria em formato JSONL (JSON Lines).

    Cada decisão de crédito é registrada em uma linha isolada,
    facilitando ingestão em ferramentas de análise.
    """

    def __init__(self) -> None:
        self.audit_dir = Path(settings.logging_audit_path)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = (
            self.audit_dir
            / f"decisions_{datetime.now(timezone.utc).strftime('%Y%m')}.jsonl"
        )

    def log_decision(
        self,
        request_id: str,
        decision: str,
        probability: float,
        threshold: float,
        top_factors: list[dict],
        metadata: dict | None = None,
    ) -> None:
        """
        Registra uma decisão de crédito no audit trail.

        Args:
            request_id: Identificador único da requisição.
            decision: "APROVADO" ou "NEGADO".
            probability: Probabilidade de inadimplência.
            threshold: Threshold utilizado.
            top_factors: Top features SHAP.
            metadata: Dados adicionais opcionais.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "decision": decision,
            "probability": round(probability, 6),
            "threshold": threshold,
            "top_factors": top_factors,
            "model_version": "1.0.0",
            "metadata": metadata or {},
        }

        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug(f"Decisão registrada: {request_id} → {decision}")
        except Exception as e:
            logger.error(f"Falha ao registrar decisão no audit trail: {e}")
