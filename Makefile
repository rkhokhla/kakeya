.PHONY: help build test clean docker-up docker-down backend-test agent-test

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Backend targets
backend-build: ## Build Go backend binary
	cd backend && go build -o server ./cmd/server

backend-test: ## Run Go backend tests
	cd backend && go test -v -cover ./...

backend-run: ## Run Go backend locally
	cd backend && go run ./cmd/server

backend-tidy: ## Tidy Go dependencies
	cd backend && go mod tidy

# Agent targets
agent-install: ## Install Python agent dependencies
	cd agent && pip install -r requirements.txt

agent-test: ## Run Python agent tests
	cd agent && pytest tests/ -v

agent-test-cov: ## Run Python agent tests with coverage
	cd agent && pytest --cov=src tests/

# Docker targets
docker-build: ## Build Docker image for backend
	cd backend && docker build -t fractal-lba/backend:latest .

docker-up: ## Start all services with Docker Compose
	cd deployments/docker && docker-compose up -d

docker-down: ## Stop all services
	cd deployments/docker && docker-compose down

docker-logs: ## Follow Docker Compose logs
	cd deployments/docker && docker-compose logs -f

docker-clean: ## Stop and remove all containers and volumes
	cd deployments/docker && docker-compose down -v

# Test targets
test: backend-test agent-test ## Run all tests

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	cd tests/integration && go test -v ./...

# Build targets
build: backend-build ## Build all components

clean: ## Clean build artifacts
	rm -f backend/server
	rm -rf agent/src/__pycache__
	rm -rf agent/**/__pycache__
	rm -rf data/

# Kubernetes targets
k8s-install: ## Install Helm chart
	helm install fractal-lba deployments/k8s/helm/fractal-lba

k8s-upgrade: ## Upgrade Helm chart
	helm upgrade fractal-lba deployments/k8s/helm/fractal-lba

k8s-uninstall: ## Uninstall Helm chart
	helm uninstall fractal-lba

k8s-template: ## Generate Kubernetes manifests
	helm template fractal-lba deployments/k8s/helm/fractal-lba

# Development targets
dev-setup: backend-tidy agent-install ## Setup development environment
	@echo "Development environment ready!"

fmt: ## Format code
	cd backend && go fmt ./...
	cd agent && black src/ tests/

lint: ## Lint code
	cd backend && golangci-lint run
	cd agent && pylint src/

# Metrics targets
metrics: ## Open Prometheus (requires Docker Compose running)
	@echo "Opening Prometheus at http://localhost:9090"
	@open http://localhost:9090 || xdg-open http://localhost:9090

grafana: ## Open Grafana (requires Docker Compose running)
	@echo "Opening Grafana at http://localhost:3000"
	@open http://localhost:3000 || xdg-open http://localhost:3000
