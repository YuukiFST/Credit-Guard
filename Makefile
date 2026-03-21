# Makefile

.PHONY: help install dev lint typecheck test train api app docker-up docker-down clean

help:  ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependências de produção
	uv pip install -r requirements.txt

dev:  ## Instala dependências de desenvolvimento e hooks
	uv pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

lint:  ## Executa linting e formatação
	ruff check src/ tests/ --fix
	ruff format src/ tests/

typecheck:  ## Verificação de tipos com mypy
	mypy src/ --ignore-missing-imports

test:  ## Executa testes com cobertura
	pytest --cov=src --cov-report=term-missing --cov-fail-under=75

train:  ## Treina e compara modelos
	python -m src.models.trainer

api:  ## Inicia a API FastAPI em modo dev
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

app:  ## Inicia o Streamlit
	streamlit run app/streamlit_app.py

docker-up:  ## Sobe todos os serviços via Docker Compose
	docker-compose up -d --build

docker-down:  ## Para todos os serviços
	docker-compose down

clean:  ## Remove artefatos de build e cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
