.PHONY: setup demo-setup demo build-frontend lint test test-slow test-frontend test-all e2e audit perf fresh-clone dev-backend dev-frontend study report clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
# npm on some macOS setups stalls on IPv6 DNS; this flag avoids it and is harmless elsewhere.
export NODE_OPTIONS ?= --no-network-family-autoselection

setup:
	python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -c requirements.lock -e ".[dev]"
	cd frontend && npm ci

# Install the committed real demo data (10 trained actors, study results, replays) into results/.
demo-setup:
	scripts/demo_setup.sh

build-frontend:
	cd frontend && npm run build

# One command: demo data + production frontend + backend. http://localhost:5173
demo: demo-setup build-frontend
	scripts/demo.sh

lint:
	$(VENV)/bin/ruff check marl backend tests scripts
	$(VENV)/bin/mypy marl/utils marl/envs marl/algorithms marl/evaluation marl/analysis marl/replay marl/networks marl/buffers marl/training marl/experiments

# Fast suite (includes the real-server integration tests; needs demo_data/, which is committed).
test:
	$(VENV)/bin/pytest -m "not slow" --cov=marl --cov=backend --cov-report=term-missing

test-slow:
	$(VENV)/bin/pytest -m slow -v

test-frontend:
	cd frontend && npm test && npm run lint && npm run build

# Browser end-to-end run (production build, demo bundle only). Writes docs/E2E_REPORT.md.
e2e: build-frontend
	$(PY) frontend/e2e/run_e2e.py

# Every number traced to a results file. Writes docs/REAL_DATA_AUDIT.md.
audit:
	$(PY) scripts/audit_real_data.py

perf:
	$(PY) scripts/perf_check.py

test-all: lint test test-slow test-frontend e2e audit

# Clone (or export the working tree), follow the README, run everything. Writes docs/FRESH_CLONE_REPORT.md.
fresh-clone:
	scripts/fresh_clone_check.sh

dev-backend:
	$(VENV)/bin/uvicorn backend.app.main:app --reload --port 8000 --ws-max-size 65536

dev-frontend:
	cd frontend && npm run dev

PARALLEL ?= 4

study:
	$(PY) scripts/run_study.py --study $(STUDY) --parallel $(PARALLEL)

report:
	$(PY) scripts/aggregate_results.py --study $(STUDY)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
