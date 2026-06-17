.PHONY: help install dev test lint clean demo train-rl train-tft docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

dev: ## Start development stack (backend + dashboard + redis + postgres)
	docker-compose up -d redis postgres
	@echo "Waiting for services..."
	@sleep 3
	uvicorn core.api.main:app --host 0.0.0.0 --port 8000 --reload &
	cd dashboard && npm run dev

test: ## Run tests
	pytest tests/ -v

lint: ## Run linter
	ruff check core/ tests/
	ruff format --check core/ tests/

format: ## Format code
	ruff format core/ tests/

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache

demo: ## Start full demo stack
	./scripts/run_demo.sh

train-rl: ## Train RL agent
	python scripts/train_rl_agent.py

train-tft: ## Train TFT forecaster
	python scripts/train_tft.py

docker-up: ## Start all services via Docker
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down
