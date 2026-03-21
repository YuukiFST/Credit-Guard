"""
Fixtures compartilhadas para testes.

Centraliza criação de objetos de teste reutilizáveis.
"""

import pandas as pd
import pytest
import numpy as np

from src.data.schemas import CreditClientInput


@pytest.fixture
def sample_client_data() -> CreditClientInput:
    """Dados válidos de um cliente para testes."""
    return CreditClientInput(
        limit_balance=50000.0,
        age=35,
        sex=2,
        education=2,
        marriage=1,
        pay_history=[-1, -1, 0, 0, 1, 0],
        bill_amounts=[23000.0, 18500.0, 21000.0, 15000.0, 17000.0, 20000.0],
        pay_amounts=[1500.0, 2000.0, 800.0, 1000.0, 500.0, 1200.0],
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame de exemplo no formato do dataset UCI."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "ID": range(1, n + 1),
        "LIMIT_BAL": np.random.randint(10000, 500000, n),
        "SEX": np.random.choice([1, 2], n),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n),
        "MARRIAGE": np.random.choice([0, 1, 2, 3], n),
        "AGE": np.random.randint(21, 70, n),
        **{f"PAY_{i}": np.random.choice([-2, -1, 0, 1, 2], n) for i in [0, 2, 3, 4, 5, 6]},
        **{f"BILL_AMT{i}": np.random.uniform(-5000, 50000, n) for i in range(1, 7)},
        **{f"PAY_AMT{i}": np.random.uniform(0, 20000, n) for i in range(1, 7)},
        "default.payment.next.month": np.random.choice([0, 1], n, p=[0.78, 0.22]),
    })
