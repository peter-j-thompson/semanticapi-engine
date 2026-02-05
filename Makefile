.PHONY: setup run test lint clean docker

# Setup development environment
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e ".[dev]"
	@echo ""
	@echo "✅ Setup complete! Activate with: source .venv/bin/activate"

# Run the server
run:
	.venv/bin/uvicorn semanticapi.server:app --host 0.0.0.0 --port 8080 --reload

# Run tests
test:
	.venv/bin/pytest tests/ -v

# Run tests with coverage
test-cov:
	.venv/bin/pytest tests/ -v --cov=semanticapi --cov-report=html

# Lint (if ruff is installed)
lint:
	.venv/bin/ruff check semanticapi/ tests/

# Build Docker image
docker:
	docker compose up --build

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	rm -rf dist build *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Show loaded providers
providers:
	@python3 -c "from semanticapi.provider_loader import load_providers; [print(f'  {p.provider_id}: {p.name} ({len(p.capabilities)} caps)') for p in load_providers()]"

# Quick health check
health:
	curl -s http://localhost:8080/health | python3 -m json.tool
