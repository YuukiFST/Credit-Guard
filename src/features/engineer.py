"""
Pipeline de feature engineering para o dataset UCI Credit Card.

Responsabilidades:
- Remover colunas irrelevantes (ID)
- Criar features derivadas (utilização, tendência, capacidade de pagamento)
- Normalizar features categóricas com encoding mínimo

Design: classe stateless (não guarda estado de treino).
Para features que precisam de fit (ex: StandardScaler), usar
o pipeline do scikit-learn que evita data leakage.
"""

import numpy as np
import pandas as pd
from loguru import logger


def align_engineered_to_model(model: object, df: pd.DataFrame) -> pd.DataFrame:
    """
    Reordena colunas para coincidir com `feature_names_in_` do modelo (treino).

    Evita erro de mismatch quando o DataFrame tem as mesmas features noutra ordem.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return df
    order = list(names)
    missing = set(order) - set(df.columns)
    if missing:
        raise KeyError(
            f"Features esperadas pelo modelo em falta: {sorted(missing)}. "
            f"Colunas atuais: {sorted(df.columns)}"
        )
    return df.loc[:, order]


class FeatureEngineer:
    """
    Pipeline de feature engineering para o dataset UCI Credit Card.

    Cria features derivadas que capturam comportamento financeiro
    mais profundo que as features originais.
    """

    # Colunas a remover — centralizadas, não hardcoded na função
    COLUMNS_TO_DROP: list[str] = ["ID"]
    TARGET_COLUMN: str = "default.payment.next.month"
    # UCI dataset tem PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 (sem PAY_1!)
    PAY_COLUMNS: list[str] = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    MONTHS: int = 6

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica feature engineering no DataFrame.

        Args:
            df: DataFrame bruto no formato do dataset UCI.

        Returns:
            DataFrame com features originais + derivadas, sem colunas desnecessárias.

        Raises:
            KeyError: Se colunas obrigatórias estiverem faltando no DataFrame.
        """
        self._validate_input_columns(df)
        result = df.copy()

        result = self._drop_unnecessary_columns(result)
        result = self._add_credit_utilization(result)
        result = self._add_utilization_growth(result)
        result = self._add_payment_ratios(result)
        result = self._add_payment_consistency(result)

        logger.debug(
            f"Feature engineering concluído: "
            f"{len(df.columns)} → {len(result.columns)} colunas"
        )
        return result

    def _validate_input_columns(self, df: pd.DataFrame) -> None:
        """Valida que todas as colunas necessárias estão presentes."""
        required = {"LIMIT_BAL", "BILL_AMT1", "PAY_AMT1", "PAY_0"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"Colunas obrigatórias faltando no DataFrame: {missing}. "
                f"Colunas presentes: {list(df.columns)}"
            )

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove colunas que não contribuem para o modelo."""
        cols_to_drop = [
            c for c in self.COLUMNS_TO_DROP + [self.TARGET_COLUMN] if c in df.columns
        ]
        return df.drop(columns=cols_to_drop)

    def _add_credit_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Razão fatura/limite — indicador de stress financeiro imediato."""
        df["CREDIT_UTILIZATION"] = (
            (df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)).clip(0, 1).fillna(0)
        )
        return df

    def _add_utilization_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tendência de crescimento do uso de crédito nos últimos 6 meses."""
        recent = df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
        older = df[["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
        df["UTILIZATION_GROWTH_6M"] = (
            ((recent - older) / df["LIMIT_BAL"].replace(0, np.nan))
            .fillna(0)
            .clip(-1, 1)
        )
        return df

    def _add_payment_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Capacidade de pagamento: quanto do total faturado foi pago."""
        for month in range(1, self.MONTHS + 1):
            bill_col = f"BILL_AMT{month}"
            pay_col = f"PAY_AMT{month}"
            ratio_col = f"PAYMENT_RATIO_{month}"
            df[ratio_col] = (
                (df[pay_col] / df[bill_col].replace(0, np.nan)).clip(0, 2).fillna(1.0)
            )  # sem fatura = pagamento completo
        return df

    def _add_payment_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Número de meses com atraso de pagamento nos últimos 6 meses."""
        pay_cols = [c for c in self.PAY_COLUMNS if c in df.columns]
        df["MONTHS_WITH_DELAY"] = (df[pay_cols] > 0).sum(axis=1)
        df["MAX_PAYMENT_DELAY"] = df[pay_cols].max(axis=1)
        return df
