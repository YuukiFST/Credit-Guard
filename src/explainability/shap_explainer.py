"""
Módulo de explicabilidade via SHAP.

Responsabilidades:
- Calcular SHAP values para instâncias individuais
- Extrair top features por impacto
- Gerar contexto estatístico para os prompts do LLM
"""

from typing import TypedDict, cast

import numpy as np
import pandas as pd
import shap


class TopFactorRow(TypedDict):
    feature: str
    shap_value: float


class SHAPExplainer:
    """
    Explainer SHAP para modelos tree-based.

    Calcula contribuição de cada feature para uma predição individual,
    permitindo explicações transparentes e auditáveis.
    """

    def __init__(self, model: object) -> None:
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcula SHAP values para uma ou mais instâncias.

        Args:
            X: DataFrame com features (1 ou mais linhas).

        Returns:
            Array de SHAP values.
        """
        shap_values = self.explainer.shap_values(X, check_additivity=False)

        # Normaliza formato: TreeExplainer pode retornar list ou array
        if isinstance(shap_values, list):
            return np.array(shap_values[1])  # classe positiva (default)
        if len(shap_values.shape) == 3:
            return shap_values[:, :, 1]  # classe positiva
        return shap_values

    def get_top_factors(
        self,
        shap_values: np.ndarray,
        feature_names: list[str] | None = None,
        n: int = 5,
    ) -> list[TopFactorRow]:
        """
        Extrai as top N features com maior impacto SHAP.

        Args:
            shap_values: SHAP values de uma instância (1D array).
            feature_names: Nomes das features. Se None, usa índices.
            n: Número de features a retornar.

        Returns:
            Lista de dicts com feature name e SHAP value.
        """
        values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        names = feature_names or [f"feature_{i}" for i in range(len(values))]

        df = pd.DataFrame(
            {
                "feature": names,
                "shap_value": values,
            }
        )
        df["abs_val"] = df["shap_value"].abs()
        top = df.sort_values("abs_val", ascending=False).head(n)

        return cast(
            list[TopFactorRow],
            top[["feature", "shap_value"]].to_dict(orient="records"),
        )

    def get_statistical_context(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        client_proba: float,
    ) -> dict:
        """
        Gera contexto estatístico do dataset para o prompt do LLM.

        Args:
            X_train: Features do dataset de treino.
            y_train: Labels do dataset de treino.
            client_proba: Probabilidade predita para o cliente.

        Returns:
            Dicionário com métricas do dataset.
        """
        default_rate = float(y_train.mean())
        # Percentil do cliente no modelo
        all_probas = self.model.predict_proba(X_train)[:, 1]  # type: ignore[attr-defined]
        client_percentile = float(
            np.percentile(
                np.searchsorted(np.sort(all_probas), client_proba)
                / len(all_probas)
                * 100,
                50,
            )
        )

        return {
            "default_rate": default_rate,
            "client_percentile": min(client_percentile, 99),
            "mean_values": X_train.mean().to_dict(),
        }
