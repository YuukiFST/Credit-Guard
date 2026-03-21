"""Testes unitários para feature engineering."""

import pandas as pd
import numpy as np
import pytest

from src.features.engineer import FeatureEngineer


class TestFeatureEngineer:
    """Testes para o pipeline de feature engineering."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_dataframe: pd.DataFrame) -> None:
        """Setup compartilhado."""
        self.fe = FeatureEngineer()
        self.df = sample_dataframe

    def test_transform_adds_new_columns(self) -> None:
        """Transform deve adicionar colunas derivadas."""
        result = self.fe.transform(self.df)
        assert "CREDIT_UTILIZATION" in result.columns
        assert "UTILIZATION_GROWTH_6M" in result.columns
        assert "MONTHS_WITH_DELAY" in result.columns

    def test_transform_removes_id(self) -> None:
        """Transform deve remover a coluna ID."""
        result = self.fe.transform(self.df)
        assert "ID" not in result.columns

    def test_transform_removes_target(self) -> None:
        """Transform deve remover a coluna target."""
        result = self.fe.transform(self.df)
        assert "default.payment.next.month" not in result.columns

    def test_credit_utilization_bounded(self) -> None:
        """Utilização de crédito deve estar entre 0 e 1."""
        result = self.fe.transform(self.df)
        assert result["CREDIT_UTILIZATION"].min() >= 0
        assert result["CREDIT_UTILIZATION"].max() <= 1

    def test_payment_ratios_created(self) -> None:
        """Payment ratios devem ser criados para todos os 6 meses."""
        result = self.fe.transform(self.df)
        for month in range(1, 7):
            assert f"PAYMENT_RATIO_{month}" in result.columns

    def test_validates_missing_columns(self) -> None:
        """Deve levantar erro se colunas obrigatórias estiverem faltando."""
        df_bad = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(KeyError, match="obrigatórias"):
            self.fe.transform(df_bad)

    def test_months_with_delay_is_integer(self) -> None:
        """MONTHS_WITH_DELAY deve ser inteiro."""
        result = self.fe.transform(self.df)
        assert result["MONTHS_WITH_DELAY"].dtype in [np.int64, np.int32, int]
