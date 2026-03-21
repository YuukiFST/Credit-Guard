"""
Guardrails de validação pós-geração do LLM.

Valida a narrativa gerada antes de retorná-la ao usuário.
Protege contra: promessas indevidas, PII, linguagem discriminatória.
"""

from loguru import logger


# Termos que NUNCA devem aparecer na narrativa gerada
BLOCKED_TERMS: list[str] = [
    "garanto",
    "garantimos",
    "certamente aprovado",
    "100% de chance",
    "discriminação",
    "raça",
    "etnia",
    "CPF",
    "nome completo",
]

# Tamanho máximo da narrativa
MAX_NARRATIVE_LENGTH: int = 1500


def validate_narrative(
    narrative: str,
    expected_factors: list[str],
    decision: str,
) -> tuple[bool, str]:
    """
    Valida a narrativa gerada pelo LLM.

    Args:
        narrative: Texto gerado pelo LLM.
        expected_factors: Features SHAP que deveriam estar mencionadas.
        decision: Decisão real ("APROVADO" ou "NEGADO").

    Returns:
        Tupla (is_valid, reason).
    """
    narrative_lower = narrative.lower()

    # 1. Termos proibidos
    for term in BLOCKED_TERMS:
        if term.lower() in narrative_lower:
            logger.warning(f"Guardrail: termo proibido '{term}' encontrado")
            return False, f"Narrativa contém termo proibido: '{term}'"

    # 2. Consistência com decisão
    if decision == "NEGADO" and "foi aprovado" in narrative_lower:
        logger.warning("Guardrail: narrativa inconsistente (diz aprovado, decisão negado)")
        return False, "Narrativa inconsistente com a decisão"
    if decision == "APROVADO" and "foi negado" in narrative_lower:
        logger.warning("Guardrail: narrativa inconsistente (diz negado, decisão aprovado)")
        return False, "Narrativa inconsistente com a decisão"

    # 3. Tamanho
    if len(narrative) > MAX_NARRATIVE_LENGTH:
        logger.info(f"Guardrail: narrativa excede tamanho máximo ({len(narrative)} chars)")
        return False, "Narrativa excede tamanho máximo"

    return True, "OK"


def safe_generate(
    narrative: str,
    expected_factors: list[str],
    decision: str,
    fallback_message: str = "Explicação não disponível. Consulte os fatores listados acima.",
) -> str:
    """
    Retorna a narrativa se válida, ou mensagem de fallback se inválida.

    Nunca propaga conteúdo inadequado ao usuário final.
    """
    is_valid, reason = validate_narrative(narrative, expected_factors, decision)
    if is_valid:
        return narrative
    logger.error(f"Narrativa rejeitada pelo guardrail: {reason}")
    return fallback_message
