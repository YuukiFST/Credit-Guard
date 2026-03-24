"""
Router de health check.

Endpoint: GET /health
Usado por Docker HEALTHCHECK e load balancers.
"""

from datetime import UTC, datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check() -> dict:
    """Retorna status do serviço e timestamp para monitoramento."""
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
    }
