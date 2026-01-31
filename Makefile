.PHONY: install dev run test lint clean help

# Default target
help:
	@echo "Retail Forecast Dashboard - Available Commands"
	@echo "=============================================="
	@echo "make install   - Install production dependencies"
	@echo "make dev       - Install with dev dependencies"
	@echo "make run       - Start the API server"
	@echo "make test      - Run all tests"
	@echo "make lint      - Run code linting"
	@echo "make clean     - Remove cache files"
	@echo ""
	@echo "Quick start: make install && make run"

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

run:
	@echo "Starting API server at http://127.0.0.1:8000"
	@echo "Dashboard: http://127.0.0.1:8000/dashboard"
	@echo "API Docs:  http://127.0.0.1:8000/docs"
	@echo ""
	python -m uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned cache files"
