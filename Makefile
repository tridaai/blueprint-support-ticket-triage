# Trida AI Blueprint — Support Ticket Triage Agent
#
# Primary workflow is uv (https://docs.astral.sh/uv/):
#   make install     uv sync --extra dev
#   make demo        one ticket end to end, offline (no keys needed)
#   make test        pytest
#   make evals       golden routing gate + calibration gate (both offline)
#   make serve       the API + console on http://localhost:8000
#
# No uv? python3 -m venv .venv && source .venv/bin/activate, then:
#   make install-pip pip install -e ".[dev]" -c constraints.txt

.PHONY: install install-pip demo test evals serve

install:
	uv sync --extra dev

install-pip:
	pip install -e ".[dev]" -c constraints.txt

demo:
	uv run support-triage-demo

test:
	uv run pytest -q

evals:
	uv run python evals/run_evals.py
	uv run python evals/run_calibration.py

serve:
	uv run uvicorn support_triage.api:app --port 8000
