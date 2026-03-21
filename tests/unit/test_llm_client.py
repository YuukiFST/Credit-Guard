"""Testes unitários para o cliente LLM com fallback."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.client import LLMClientWithFallback


class TestLLMClientWithFallback:
    """Testes para o mecanismo de fallback do LLM."""

    @pytest.mark.asyncio
    async def test_uses_ollama_when_available(self) -> None:
        """Deve usar Ollama quando disponível."""
        client = LLMClientWithFallback()
        client.primary.is_available = MagicMock(return_value=True)
        client.primary.generate = AsyncMock(return_value="Narrativa do Ollama")

        result = await client.generate("prompt", "system")
        assert result == "Narrativa do Ollama"

    @pytest.mark.asyncio
    async def test_falls_back_to_groq(self) -> None:
        """Deve usar Groq quando Ollama falha."""
        client = LLMClientWithFallback()
        client.primary.is_available = MagicMock(return_value=False)
        client.fallback.is_available = MagicMock(return_value=True)
        client.fallback.generate = AsyncMock(return_value="Narrativa do Groq")

        result = await client.generate("prompt", "system")
        assert result == "Narrativa do Groq"

    @pytest.mark.asyncio
    async def test_returns_default_when_all_fail(self) -> None:
        """Deve retornar mensagem padrão quando ambos falham."""
        client = LLMClientWithFallback()
        client.primary.is_available = MagicMock(return_value=False)
        client.fallback.is_available = MagicMock(return_value=False)

        result = await client.generate("prompt", "system")
        assert result == client.DEFAULT_MESSAGE

    @pytest.mark.asyncio
    async def test_never_raises_exception(self) -> None:
        """Nunca deve levantar exceção, mesmo com erros em is_available."""
        client = LLMClientWithFallback()
        client.primary.is_available = MagicMock(side_effect=ConnectionError("nope"))
        client.fallback.is_available = MagicMock(side_effect=RuntimeError("nope"))

        result = await client.generate("prompt", "system")
        assert result == client.DEFAULT_MESSAGE
