"""
Monitoramento de Model Drift via PSI (Population Stability Index).

Padrão da indústria bancária para detectar degradação de modelos.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """
    Calcula o Population Stability Index (PSI).

    Interpretação:
    - PSI < 0.1: sem mudança significativa → modelo estável
    - 0.1 ≤ PSI < 0.2: mudança moderada → investigar
    - PSI ≥ 0.2: mudança significativa → retreinar modelo

    Args:
        expected: Distribuição de referência (treino).
        actual: Distribuição observada (produção).
        buckets: Número de bins.

    Returns:
        Valor do PSI.
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return float(psi)


def check_feature_drift(
    train_df: pd.DataFrame,
    production_df: pd.DataFrame,
    threshold: float = 0.2,
) -> dict[str, dict]:
    """
    Verifica drift em todas as features numéricas.

    Args:
        train_df: DataFrame de treino (referência).
        production_df: DataFrame de produção.
        threshold: Limite de PSI para alerta.

    Returns:
        Dicionário com PSI de cada feature e status.
    """
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    results: dict[str, dict] = {}

    for col in numeric_cols:
        if col in production_df.columns:
            psi = calculate_psi(
                train_df[col].dropna().values,
                production_df[col].dropna().values,
            )
            status = (
                "✅ Estável" if psi < 0.1
                else "⚠️ Investigar" if psi < threshold
                else "🚨 Retreinar"
            )
            results[col] = {"psi": round(psi, 4), "status": status}

            if psi >= threshold:
                logger.warning(f"DRIFT detectado em {col}: PSI={psi:.4f} ({status})")

    return results
