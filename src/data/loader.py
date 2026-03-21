"""
Módulo de carregamento e preparação inicial de dados.

Responsabilidades:
- Carregar o dataset UCI Credit Card
- Split treino/teste estratificado
- Carregar dados de teste para validação de modelo no CI
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

from src.config import settings


def load_raw_data() -> pd.DataFrame:
    """
    Carrega o dataset UCI Credit Card do caminho configurado.

    Returns:
        DataFrame bruto com todas as colunas originais.

    Raises:
        FileNotFoundError: Se o arquivo CSV não existir no caminho configurado.
    """
    path = settings.data_raw_path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. "
            f"Verifique se o arquivo UCI_Credit_Card.csv está em data/raw/."
        )
    df = pd.read_csv(path)
    logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide o dataset em treino e teste com estratificação.

    Args:
        df: DataFrame completo.
        target_column: Nome da coluna target. Default do config.
        test_size: Proporção do test set. Default do config.
        random_state: Seed para reprodutibilidade. Default do config.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    target = target_column or settings.data_target_column
    size = test_size or settings.data_test_size
    seed = random_state or settings.data_random_state

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=size,
        random_state=seed,
        stratify=y,
    )

    logger.info(
        f"Split concluído: treino={len(X_train)}, teste={len(X_test)} "
        f"(positivos: treino={y_train.mean():.1%}, teste={y_test.mean():.1%})"
    )
    return X_train, X_test, y_train, y_test


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega dados de teste para validação de modelo no CI.

    Usado pelo pipeline de CI/CD para verificar ROC-AUC mínimo.

    Returns:
        Tupla (X_test, y_test).
    """
    df = load_raw_data()
    _, X_test, _, y_test = split_data(df)
    return X_test, y_test
