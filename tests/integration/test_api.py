"""Testes de integração da API FastAPI."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


class TestHealthEndpoint:
    """Testes para o endpoint /health."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self) -> None:
        """Health check deve retornar status ok."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "version" in data
