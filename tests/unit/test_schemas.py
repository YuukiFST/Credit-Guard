"""Testes unitários para schemas Pydantic."""

import pytest

from src.data.schemas import CreditClientInput


class TestCreditClientInput:
    """Testes para validação de dados de entrada."""

    def test_valid_input(self, sample_client_data: CreditClientInput) -> None:
        """Dados válidos devem criar o schema sem erros."""
        assert sample_client_data.limit_balance == 50000.0
        assert sample_client_data.age == 35

    def test_negative_limit_balance_rejected(self) -> None:
        """Limite de crédito negativo deve ser rejeitado."""
        with pytest.raises(ValueError, match="greater than"):
            CreditClientInput(
                limit_balance=-1000.0,
                age=35,
                sex=2,
                education=2,
                marriage=1,
                pay_history=[-1, -1, 0, 0, 1, 0],
                bill_amounts=[0.0] * 6,
                pay_amounts=[0.0] * 6,
            )

    def test_invalid_age_rejected(self) -> None:
        """Idade fora do range deve ser rejeitada."""
        with pytest.raises(ValueError):
            CreditClientInput(
                limit_balance=50000.0,
                age=10,
                sex=2,
                education=2,
                marriage=1,
                pay_history=[-1, -1, 0, 0, 1, 0],
                bill_amounts=[0.0] * 6,
                pay_amounts=[0.0] * 6,
            )

    def test_invalid_pay_history_value(self) -> None:
        """Valores inválidos em pay_history devem ser rejeitados."""
        with pytest.raises(ValueError, match="inválidos"):
            CreditClientInput(
                limit_balance=50000.0,
                age=35,
                sex=2,
                education=2,
                marriage=1,
                pay_history=[99, -1, 0, 0, 1, 0],
                bill_amounts=[0.0] * 6,
                pay_amounts=[0.0] * 6,
            )

    def test_wrong_list_length(self) -> None:
        """Listas com tamanho errado devem ser rejeitadas."""
        with pytest.raises(ValueError):
            CreditClientInput(
                limit_balance=50000.0,
                age=35,
                sex=2,
                education=2,
                marriage=1,
                pay_history=[-1, -1, 0],  # apenas 3 meses
                bill_amounts=[0.0] * 6,
                pay_amounts=[0.0] * 6,
            )

    def test_to_dataframe(self, sample_client_data: CreditClientInput) -> None:
        """Conversão para DataFrame deve manter todas as colunas."""
        df = sample_client_data.to_dataframe()
        assert len(df) == 1
        assert "LIMIT_BAL" in df.columns
        assert "PAY_0" in df.columns
        assert "PAY_2" in df.columns
        assert "PAY_1" not in df.columns
        assert "BILL_AMT6" in df.columns
        assert "PAY_AMT6" in df.columns
