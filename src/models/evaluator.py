"""
Módulo de avaliação de modelos com métricas técnicas e de negócio.

Responsabilidades:
- Calcular métricas completas (ROC-AUC, F1, KS, Gini)
- Encontrar threshold ótimo via análise custo-benefício
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import ks_2samp


def calculate_full_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula métricas completas para avaliação do modelo.

    Args:
        y_true: Labels reais.
        y_proba: Probabilidades preditas de inadimplência.
        threshold: Limiar de classificação.

    Returns:
        Dicionário com todas as métricas.
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    gini = 2 * roc_auc - 1

    # KS Statistic
    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    ks_stat = ks_2samp(pos_proba, neg_proba).statistic if len(pos_proba) > 0 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc),
        "gini": float(gini),
        "ks_statistic": float(ks_stat),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_false_positive: float,
    cost_false_negative: float,
) -> dict[str, float | list]:
    """
    Encontra o threshold que minimiza o custo total de negócio.

    Args:
        y_true: Labels reais.
        y_proba: Probabilidades preditas.
        cost_false_positive: Custo de negar crédito a bom pagador (R$).
        cost_false_negative: Custo de conceder crédito a inadimplente (R$).

    Returns:
        Dicionário com threshold ótimo, custo mínimo e curva de custo.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    costs: list[float] = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
        costs.append(float(total_cost))

    optimal_idx = int(np.argmin(costs))
    return {
        "optimal_threshold": float(thresholds[optimal_idx]),
        "min_total_cost": costs[optimal_idx],
        "threshold_curve": {
            "thresholds": thresholds.tolist(),
            "costs": costs,
        },
    }
