"""
Calibração de probabilidades do modelo.

Após calibração, predict_proba() retorna probabilidades que
refletem a frequência real de eventos no dataset.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(
    model: object,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "isotonic",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Calibra as probabilidades do modelo usando validação cruzada.

    Args:
        model: Modelo treinado.
        X_val: Features do conjunto de validação.
        y_val: Labels do conjunto de validação.
        method: "isotonic" ou "sigmoid" (Platt scaling).
        cv: Número de folds para calibração.

    Returns:
        Modelo calibrado.
    """
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
    )
    calibrated.fit(X_val, y_val)

    y_proba_before = model.predict_proba(X_val)[:, 1]  # type: ignore[union-attr]
    y_proba_after = calibrated.predict_proba(X_val)[:, 1]

    brier_before = float(np.mean((y_proba_before - y_val) ** 2))
    brier_after = float(np.mean((y_proba_after - y_val) ** 2))

    logger.info(
        f"Calibração ({method}): Brier Score {brier_before:.4f} → {brier_after:.4f} "
        f"({'melhorou' if brier_after < brier_before else 'piorou'})"
    )
    return calibrated
