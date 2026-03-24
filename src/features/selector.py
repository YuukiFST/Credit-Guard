"""
Seletor de features baseado em importância SHAP global.

Remove features com impacto negligenciável e features
altamente correlacionadas (redundantes), melhorando
performance e interpretabilidade do modelo.
"""

import numpy as np
import pandas as pd
import shap
from loguru import logger


class FeatureSelector:
    """
    Seletor de features baseado em importância SHAP global.

    Remove features com impacto negligenciável e features
    altamente correlacionadas (redundantes).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        min_shap_importance: float = 0.01,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.min_shap_importance = min_shap_importance
        self.selected_features: list[str] = []

    def select(
        self,
        X: pd.DataFrame,
        model: object,
        shap_sample_size: int = 500,
    ) -> list[str]:
        """
        Seleciona features relevantes usando SHAP e análise de correlação.

        Args:
            X: DataFrame com todas as features.
            model: Modelo treinado para cálculo do SHAP.
            shap_sample_size: Amostras para cálculo SHAP (economia de tempo).

        Returns:
            Lista de nomes das features selecionadas.
        """
        # 1. Importância via SHAP
        sample = X.sample(min(shap_sample_size, len(X)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # classe positiva

        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        # 2. Filtra por importância mínima
        relevant = importance_df[
            importance_df["importance"] >= self.min_shap_importance
        ]
        logger.info(
            f"Features relevantes (SHAP >= {self.min_shap_importance}): "
            f"{len(relevant)} de {len(importance_df)}"
        )

        # 3. Remove correlações altas
        corr_matrix = X[relevant["feature"]].corr().abs()
        to_drop: set[str] = set()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    feat_i = corr_matrix.columns[i]
                    feat_j = corr_matrix.columns[j]
                    imp_i = importance_df.set_index("feature").loc[feat_i, "importance"]
                    imp_j = importance_df.set_index("feature").loc[feat_j, "importance"]
                    drop = feat_j if imp_i > imp_j else feat_i
                    to_drop.add(drop)
                    logger.debug(
                        f"Removendo {drop} (correlação {corr_matrix.iloc[i, j]:.2f})"
                    )

        self.selected_features = [f for f in relevant["feature"] if f not in to_drop]
        logger.info(
            f"Features finais: {len(self.selected_features)} "
            f"({len(to_drop)} removidas por correlação)"
        )
        return self.selected_features
