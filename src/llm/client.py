"""
Cliente LLM com fallback automático.

Arquitetura: Ollama local → Groq API → mensagem padrão.
Nunca quebra a aplicação, mesmo sem LLM disponível.
"""

import httpx
from loguru import logger

from src.config import settings


class OllamaClient:
    """Cliente para Ollama local."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url
        self.model = settings.llm_local_model

    def is_available(self) -> bool:
        """Verifica se Ollama está rodando."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Gera texto via Ollama."""
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": settings.llm_max_tokens},
                },
            )
            response.raise_for_status()
            return response.json()["response"]


class GroqClient:
    """Cliente para Groq API (fallback)."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self) -> None:
        self.model = settings.llm_fallback_model
        self.api_key = settings.groq_api_key

    def is_available(self) -> bool:
        """Verifica se a API key está configurada."""
        return bool(self.api_key and self.api_key != "gsk_sua_chave_aqui")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Gera texto via Groq API."""
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            response = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": settings.llm_max_tokens,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


class LLMClientWithFallback:
    """
    Orquestrador LLM com fallback automático.

    Ordem: Ollama local → Groq API → mensagem padrão.
    """

    DEFAULT_MESSAGE = (
        "Explicação por IA não disponível no momento. "
        "Consulte os fatores SHAP listados para compreender a decisão."
    )

    def __init__(self) -> None:
        self.primary = OllamaClient()
        self.fallback = GroqClient()

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Gera texto com fallback automático entre backends LLM.

        Nunca levanta exceção — retorna mensagem padrão em caso de falha total.
        """
        # Tenta Ollama primeiro
        try:
            primary_available = self.primary.is_available()
        except Exception as e:
            logger.warning(f"Erro ao verificar disponibilidade do Ollama: {e}")
            primary_available = False

        if primary_available:
            try:
                logger.debug("Usando Ollama local para geração de narrativa.")
                return await self.primary.generate(prompt, system_prompt)
            except Exception as e:
                logger.warning(f"Ollama falhou: {e}. Tentando Groq como fallback.")

        # Fallback para Groq
        try:
            fallback_available = self.fallback.is_available()
        except Exception as e:
            logger.warning(f"Erro ao verificar disponibilidade do Groq: {e}")
            fallback_available = False

        if fallback_available:
            try:
                logger.info("Usando Groq API como fallback para LLM.")
                return await self.fallback.generate(prompt, system_prompt)
            except Exception as e:
                logger.error(f"Groq também falhou: {e}. Retornando mensagem padrão.")

        logger.warning("Nenhum backend LLM disponível. Retornando mensagem padrão.")
        return self.DEFAULT_MESSAGE
