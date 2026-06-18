.PHONY: help install dev test lint clean demo

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

dev: ## Start Streamlit dashboard
	streamlit run app.py

test: ## Run tests
	pytest tests/ -v

lint: ## Run linter
	ruff check core/ tests/ app.py
	ruff format --check core/ tests/ app.py

format: ## Format code
	ruff format core/ tests/ app.py

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache

demo: ## Run full demo pipeline
	python scripts/demo_vision.py --source 0 --show --violations --signal RED
