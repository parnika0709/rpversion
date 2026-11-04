.PHONY: help install api web ml lint test clean

help:
	@echo "Brain Tumor MRI — monorepo"
	@echo "  make install   Install ML + API packages in editable mode"
	@echo "  make api       Run FastAPI backend (uvicorn)"
	@echo "  make web       Run React frontend (Vite dev)"
	@echo "  make ml        Run ML training (placeholder; see packages/ml/)"
	@echo "  make lint      Run ruff + mypy"
	@echo "  make test      Run pytest"
	@echo "  make clean     Remove caches"

install:
	pip install -e packages/ml
	pip install -e apps/api

api:
	uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000

web:
	cd apps/web && npm install && npm run dev

lint:
	ruff check packages/ml apps/api
	mypy packages/ml apps/api

test:
	pytest -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
