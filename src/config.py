"""
Módulo de configuração central do projeto.

Carrega configurações do config.yaml e variáveis de ambiente.
Todas as outras partes do código importam daqui — nunca hardcode valores.
"""

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.parent


def _load_yaml(path: Path) -> dict:
    """Carrega o arquivo YAML de configuração."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Carrega config.yaml uma única vez na importação do módulo
_yaml_config = _load_yaml(ROOT_DIR / "config.yaml")


class Settings(BaseSettings):
    """
    Configurações da aplicação com validação via Pydantic.

    Variáveis de ambiente sobrescrevem os defaults.
    Variáveis sensíveis (ex: GROQ_API_KEY) são carregadas APENAS do .env.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),   # permite campos model_*
        extra="ignore",            # ignora env vars não mapeadas
    )

    # Variáveis sensíveis — OBRIGATÓRIAS no .env (nunca no config.yaml)
    groq_api_key: str = Field(default="", description="Chave da API Groq para fallback LLM")

    # --- Data ---
    data_raw_path: Path = Field(
        default=ROOT_DIR / _yaml_config["data"]["raw_path"]
    )
    data_target_column: str = Field(
        default=_yaml_config["data"]["target_column"]
    )
    data_test_size: float = Field(
        default=_yaml_config["data"]["test_size"]
    )
    data_random_state: int = Field(
        default=_yaml_config["data"]["random_state"]
    )

    # --- Model ---
    model_default_threshold: float = Field(
        default=_yaml_config["model"]["default_threshold"]
    )
    model_cv_folds: int = Field(
        default=_yaml_config["model"]["cv_folds"]
    )
    model_optuna_trials: int = Field(
        default=_yaml_config["model"]["optuna_trials"]
    )
    model_artifact_path: Path = Field(
        default=ROOT_DIR / _yaml_config["model"]["artifact_path"]
    )

    # --- Business ---
    business_cost_fp: float = Field(
        default=_yaml_config["business"]["cost_false_positive"]
    )
    business_cost_fn: float = Field(
        default=_yaml_config["business"]["cost_false_negative"]
    )

    # --- LLM ---
    llm_local_model: str = Field(
        default=_yaml_config["llm"]["local_model"]
    )
    llm_fallback_model: str = Field(
        default=_yaml_config["llm"]["fallback_model"]
    )
    llm_timeout: int = Field(
        default=_yaml_config["llm"]["timeout_seconds"]
    )
    llm_max_tokens: int = Field(
        default=_yaml_config["llm"]["max_tokens"]
    )
    llm_prompt_version: str = Field(
        default=_yaml_config["llm"]["prompt_version"]
    )

    # --- Logging ---
    logging_level: str = Field(
        default=_yaml_config["logging"]["level"]
    )
    logging_audit_path: str = Field(
        default=_yaml_config["logging"]["audit_path"]
    )

    # --- API ---
    api_host: str = Field(
        default=_yaml_config["api"]["host"]
    )
    api_port: int = Field(
        default=_yaml_config["api"]["port"]
    )


# Instância global — importar esta em todo o projeto
settings = Settings()
