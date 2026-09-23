.PHONY: setup lint test test-slow dev-backend dev-frontend study report clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	cd frontend && npm install

lint:
	$(VENV)/bin/ruff check marl backend tests scripts
	$(VENV)/bin/mypy marl/utils marl/envs

test:
	$(VENV)/bin/pytest -m "not slow" --cov=marl --cov=backend --cov-report=term-missing

test-slow:
	$(VENV)/bin/pytest -m slow -v

dev-backend:
	$(VENV)/bin/uvicorn backend.app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

study:
	$(PY) scripts/run_study.py --study $(STUDY)

report:
	$(PY) scripts/aggregate_results.py --study $(STUDY)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
