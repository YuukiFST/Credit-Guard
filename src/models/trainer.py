"""
Módulo de treino, otimização e comparação de modelos.

Responsabilidades:
- Treinar RF, XGBoost e LightGBM com Optuna
- Registrar experimentos no MLflow
- Serializar melhor modelo
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import settings
from src.models.evaluator import calculate_full_metrics


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 50,
) -> dict[str, dict]:
    """
    Treina e compara Random Forest, XGBoost e LightGBM com Optuna.

    Args:
        X_train: Features de treino.
        y_train: Labels de treino.
        X_test: Features de teste (hold-out final).
        y_test: Labels de teste.
        n_trials: Número de tentativas do Optuna por modelo.

    Returns:
        Dicionário com resultados de cada modelo.
    """
    mlflow.set_experiment("credit_risk_model_comparison")
    results: dict[str, dict] = {}

    model_configs = {
        "random_forest": _optimize_random_forest,
        "xgboost": _optimize_xgboost,
        "lightgbm": _optimize_lightgbm,
    }

    for model_name, optimizer_fn in model_configs.items():
        logger.info(f"Otimizando e avaliando: {model_name}")
        with mlflow.start_run(run_name=model_name):
            best_params = optimizer_fn(X_train, y_train, n_trials=n_trials)
            model = _build_model(model_name, best_params)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = calculate_full_metrics(
                y_test.values, y_proba, settings.model_default_threshold
            )

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("n_optuna_trials", n_trials)
            mlflow.log_param("dataset_size", len(X_train))
            mlflow.sklearn.log_model(model, artifact_path=model_name)

            results[model_name] = {
                "model": model,
                "best_params": best_params,
                "metrics": metrics,
                "roc_auc": metrics["roc_auc"],
            }

            logger.info(
                f"{model_name} — ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"KS: {metrics['ks_statistic']:.4f}"
            )

    return results


def save_best_model(results: dict[str, dict]) -> Path:
    """
    Salva o melhor modelo baseado em ROC-AUC.

    Args:
        results: Dicionário de resultados do compare_models.

    Returns:
        Path do modelo salvo.
    """
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]
    model_path = settings.model_artifact_path / "best_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"Melhor modelo ({best_name}) salvo em {model_path}")
    return model_path


def load_model(path: Path | None = None) -> object:
    """
    Carrega modelo serializado.

    Args:
        path: Caminho do modelo. Default: models/best_model.pkl.

    Returns:
        Modelo treinado.
    """
    model_path = path or (settings.model_artifact_path / "best_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. "
            f"Execute o pipeline de treinamento primeiro."
        )
    return joblib.load(model_path)


def _build_model(name: str, params: dict) -> object:
    """Constrói modelo a partir do nome e parâmetros."""
    builders = {
        "random_forest": lambda p: RandomForestClassifier(**p, n_jobs=-1),
        "xgboost": lambda p: XGBClassifier(**p, verbosity=0, n_jobs=-1),
        "lightgbm": lambda p: LGBMClassifier(**p, n_jobs=-1, verbose=-1),
    }
    return builders[name](params)


def _optimize_random_forest(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 50,
) -> dict:
    """Otimiza hiperparâmetros do Random Forest via Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": "balanced",
            "random_state": 42,
        }
        model = RandomForestClassifier(**params, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _optimize_xgboost(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 50,
) -> dict:
    """Otimiza hiperparâmetros do XGBoost via Optuna."""
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "random_state": 42,
        }
        model = XGBClassifier(**params, verbosity=0)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _optimize_lightgbm(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 50,
) -> dict:
    """Otimiza hiperparâmetros do LightGBM via Optuna."""
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
        }
        model = LGBMClassifier(**params, verbose=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data.loader import load_raw_data
    from src.features.engineer import FeatureEngineer

    logger.info("=== Iniciando pipeline de treinamento ===")

    # 1. Carregar dados
    logger.info("Carregando dataset UCI Credit Card...")
    df = load_raw_data()
    logger.info(f"Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")

    # 2. Feature Engineering
    logger.info("Aplicando feature engineering...")
    fe = FeatureEngineer()
    y = df[settings.data_target_column]
    X = fe.transform(df)
    logger.info(f"Features criadas: {X.shape[1]} colunas")

    # 3. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.data_test_size,
        random_state=settings.data_random_state,
        stratify=y,
    )
    logger.info(f"Split: treino={len(X_train)}, teste={len(X_test)}")

    # 4. Treinar e comparar modelos
    logger.info("Iniciando comparação de modelos (RF vs XGBoost vs LightGBM)...")
    results = compare_models(
        X_train, y_train, X_test, y_test,
        n_trials=settings.model_optuna_trials,
    )

    # 5. Salvar melhor modelo
    model_path = save_best_model(results)

    # 6. Resumo final
    logger.info("=== Resultados Finais ===")
    for name, res in sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        m = res["metrics"]
        logger.info(
            f"  {name:15s} | ROC-AUC: {m['roc_auc']:.4f} | "
            f"F1: {m['f1']:.4f} | KS: {m['ks_statistic']:.4f}"
        )
    logger.info(f"Melhor modelo salvo em: {model_path}")
    logger.info("=== Pipeline concluído com sucesso! ===")

