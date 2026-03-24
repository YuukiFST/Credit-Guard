"""
Análise de viés e fairness do modelo.

Calcula métricas de fairness por grupo demográfico
para compliance com LGPD (Art. 20) e regulamentações anti-discriminação.
"""

import pandas as pd
from loguru import logger


def calculate_fairness_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_attribute: pd.Series,
    attribute_name: str,
) -> pd.DataFrame:
    """
    Calcula métricas de fairness por grupo de um atributo sensível.

    Args:
        y_true: Labels reais (0=adimplente, 1=inadimplente).
        y_pred: Predições do modelo (0 ou 1).
        sensitive_attribute: Coluna de atributo sensível.
        attribute_name: Nome do atributo para logging.

    Returns:
        DataFrame com métricas por grupo.
    """
    groups = sensitive_attribute.unique()
    results = []

    for group in sorted(groups):
        mask = sensitive_attribute == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]

        approval_rate = float((group_pred == 0).mean())
        tpr = float(
            ((group_pred == 1) & (group_true == 1)).sum()
            / max((group_true == 1).sum(), 1)
        )
        fpr = float(
            ((group_pred == 1) & (group_true == 0)).sum()
            / max((group_true == 0).sum(), 1)
        )

        results.append(
            {
                "attribute": attribute_name,
                "group": group,
                "n_samples": int(mask.sum()),
                "approval_rate": round(approval_rate, 4),
                "true_positive_rate": round(tpr, 4),
                "false_positive_rate": round(fpr, 4),
            }
        )

    df = pd.DataFrame(results)

    max_rate = df["approval_rate"].max()
    min_rate = df["approval_rate"].min()
    di_ratio = min_rate / max_rate if max_rate > 0 else 0.0

    logger.info(
        f"Fairness [{attribute_name}]: "
        f"Disparate Impact Ratio = {di_ratio:.3f} "
        f"({'✅ OK' if di_ratio >= 0.8 else '⚠️ ATENÇÃO'})"
    )

    return df
