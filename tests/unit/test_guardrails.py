"""Testes unitários para guardrails do LLM."""

from src.llm.guardrails import safe_generate, validate_narrative


class TestGuardrails:
    """Testes para validação da narrativa do LLM."""

    def test_valid_narrative_passes(self) -> None:
        """Narrativa limpa deve ser aceita."""
        narrative = "O crédito foi negado devido ao histórico de pagamento."
        is_valid, reason = validate_narrative(narrative, ["PAY_1"], "NEGADO")
        assert is_valid
        assert reason == "OK"

    def test_blocked_term_rejected(self) -> None:
        """Narrativa com termo proibido deve ser rejeitada."""
        narrative = "Garanto que da próxima vez será aprovado."
        is_valid, reason = validate_narrative(narrative, ["PAY_1"], "NEGADO")
        assert not is_valid
        assert "proibido" in reason

    def test_inconsistent_decision_rejected(self) -> None:
        """Narrativa inconsistente com a decisão deve ser rejeitada."""
        narrative = "O cliente foi aprovado com base em seu bom histórico."
        is_valid, reason = validate_narrative(narrative, ["PAY_1"], "NEGADO")
        assert not is_valid
        assert "inconsistente" in reason.lower()

    def test_too_long_narrative_rejected(self) -> None:
        """Narrativa que excede tamanho máximo deve ser rejeitada."""
        narrative = "A" * 2000
        is_valid, reason = validate_narrative(narrative, ["PAY_1"], "APROVADO")
        assert not is_valid

    def test_safe_generate_returns_fallback(self) -> None:
        """safe_generate deve retornar fallback para narrativa inválida."""
        narrative = "Garanto aprovação na próxima vez."
        result = safe_generate(narrative, ["PAY_1"], "NEGADO")
        assert "Explicação não disponível" in result

    def test_safe_generate_returns_narrative(self) -> None:
        """safe_generate deve retornar narrativa válida."""
        narrative = "O crédito foi negado devido ao histórico de atraso."
        result = safe_generate(narrative, ["PAY_1"], "NEGADO")
        assert result == narrative
