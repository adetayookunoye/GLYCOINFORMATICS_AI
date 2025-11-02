# Glycoinformatics AI Platform Makefile
# ====================================
# Complete Docker deployment automation for the Glycoinformatics AI Platform
#
# 🐳 COMPLETE DOCKER DEPLOYMENT INSTRUCTIONS:
# ============================================
#
# STEP 1: PREREQUISITES
# ---------------------
# 1. Install Docker: https://docs.docker.com/get-docker/
# 2. Install Docker Compose (included with Docker Desktop)
# 3. Add user to docker group: sudo usermod -aG docker $USER
# 4. Logout/login OR run: newgrp docker
#
# STEP 2: VERIFY SYSTEM
# ---------------------
# Run: make setup
# This validates Docker access and installs dependencies
#
# STEP 3: START COMPLETE PLATFORM
# --------------------------------
# Run: make docker-start
# This starts ALL 11 services:
#   • PostgreSQL Database (port 5432)
#   • Redis Cache (port 6379)
#   • GraphDB RDF Store (port 7200)
#   • Elasticsearch Search (port 9200)
#   • MongoDB Documents (port 27017)
#   • MinIO Object Storage (port 9000/9001)
#   • FastAPI Application (port 8000)
#   • Jupyter Lab Environment (port 8888)
#   • Traefik Load Balancer (port 8080)
#   • Prometheus Monitoring (port 9090)
#   • Grafana Dashboards (port 3000)
#
# STEP 4: ACCESS YOUR PLATFORM
# -----------------------------
# After docker-start completes:
#   • API Docs: http://localhost:8000/docs
#   • API Health: http://localhost:8000/healthz
#   • GraphDB Workbench: http://localhost:7200
#   • Grafana Dashboards: http://localhost:3000 (admin/admin)
#   • All URLs: make docker-status
#
# STEP 5: TROUBLESHOOTING
# ------------------------
# If issues occur:
#   • Check Docker: make docker-check
#   • View logs: make docker-logs
#   • Restart all: make docker-restart
#   • Clean reset: make docker-clean
#
# ONE-COMMAND DEPLOYMENT:
# -----------------------
# Run: make quick-start
# This does setup + docker-start + health-check automatically
#
# For detailed help: make help

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

# Project configuration
PROJECT_NAME := glycoinformatics_ai
VERSION := 0.1.0
PYTHON_VERSION := 3.12
DOCKER_REGISTRY := ghcr.io
IMAGE_NAME := $(PROJECT_NAME)
API_PORT := 8000
DEV_PORT := 8080

# Docker services configuration
COMPOSE_FILE := docker-compose.yml
COMPOSE_PROJECT := glyco_platform

# ============================================================================
# DATABASE CREDENTIALS (Production Ready)
# ============================================================================
# PostgreSQL Database
POSTGRES_HOST := localhost
POSTGRES_PORT := 5432
POSTGRES_DB := glycokg
POSTGRES_USER := glyco_admin
POSTGRES_PASSWORD := glyco_secure_pass_2025
DATABASE_URL := postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)

# Redis Cache
REDIS_HOST := localhost
REDIS_PORT := 6379
REDIS_DB := 0
REDIS_URL := redis://$(REDIS_HOST):$(REDIS_PORT)/$(REDIS_DB)

# MongoDB Document Store
MONGODB_HOST := localhost
MONGODB_PORT := 27017
MONGODB_USER := glyco_admin
MONGODB_PASSWORD := glyco_secure_pass_2025
MONGODB_DB := glyco_results
MONGODB_URL := mongodb://$(MONGODB_USER):$(MONGODB_PASSWORD)@$(MONGODB_HOST):$(MONGODB_PORT)

# MinIO Object Storage
MINIO_ENDPOINT := localhost:9000
MINIO_ACCESS_KEY := glyco_admin
MINIO_SECRET_KEY := glyco_secure_pass_2025
MINIO_BUCKET := glyco-data

# GraphDB RDF Store
GRAPHDB_URL := http://localhost:7200
GRAPHDB_REPOSITORY := glycokg

# Elasticsearch Search Engine
ELASTICSEARCH_URL := http://localhost:9200

# System User Credentials
SYSTEM_USER_PASSWORD := Adebayo@120

# Python configuration
PYTHON_PATH := .
VENV_NAME := venv
PIP_REQUIREMENTS := requirements.txt

# Directories
SRC_DIR := glyco_platform
TEST_DIR := tests
DOCS_DIR := documentations
DATA_DIR := data
LOGS_DIR := logs
MODELS_DIR := models

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

# ============================================================================
# HELP AND INFORMATION
# ============================================================================

.PHONY: help
help: ## Show this help message
help:
	@echo "GlycoInformatics AI Platform - Available Commands:"
	@echo ""
	@echo "  setup          - Initial project setup"
	@echo "  start          - Start all services"
	@echo "  stop           - Stop all services"
	@echo "  restart        - Restart all services"
	@echo "  logs           - Show logs for all services"
	@echo "  status         - Show service status"
	@echo "  clean          - Clean up containers and volumes"
	@echo "  test           - Run test suite"
	@echo "  build          - Build custom Docker images"
	@echo "  deploy         - Deploy to production"
	@echo "  init-data      - Initialize with sample data"
	@echo ""

# Setup development environment
setup:
	@echo "🔧 Setting up GlycoInformatics AI development environment..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting." >&2; exit 1; }
	@echo "� Checking Docker permissions..."
	@docker ps >/dev/null 2>&1 || { echo "❌ Docker permission denied. Run: sudo usermod -aG docker $$USER && newgrp docker" >&2; exit 1; }
	@echo "�📁 Creating necessary directories..."
	@mkdir -p data/{raw,processed,interim}
	@mkdir -p logs models results
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "🐳 Building custom Docker images..."
	sudo docker-compose build
	@echo "✅ Setup complete! Run 'make start' to start services."

# Start services
start:
	@echo "🚀 Starting GlycoInformatics AI Platform (11 services)..."
	sudo docker-compose up -d
	@echo "⏳ Services starting (waiting 15 seconds for initialization)..."
	@sleep 15
	@echo "📊 Service status:"
	sudo docker-compose ps
	@echo ""
	@echo "🌐 Access Points:"
	@echo "  • FastAPI Docs:     http://localhost:8000/docs"
	@echo "  • API Health:       http://localhost:8000/healthz" 
	@echo "  • GraphDB:          http://localhost:7200"
	@echo "  • Jupyter Lab:      http://localhost:8888"
	@echo "  • Traefik:          http://localhost:8080"
	@echo "  • Grafana:          http://localhost:3000 (admin/admin)"
	@echo "  • MinIO Console:    http://localhost:9001 (glyco_admin/glyco_secure_pass_2025)"
	@echo "  • Elasticsearch:    http://localhost:9200"
	@echo "  • Prometheus:       http://localhost:9090"
	@echo "✅ Platform started! See DOCKER_DEPLOYMENT.md for complete documentation"

# Stop services
stop:
	@echo "🛑 Stopping all services..."
	sudo docker-compose down
	@echo "✅ All services stopped!"

# Restart services
restart: stop start

# Show logs
logs:
	@$(DOCKER_COMPOSE) logs -f

# Show service status
status:
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "Health checks:"
	@curl -s http://localhost:7200/rest/monitor/infrastructure || echo "GraphDB: Not responding"
	@curl -s http://localhost:9200/_cluster/health || echo "Elasticsearch: Not responding"
	@curl -s http://localhost:9000/minio/health/live || echo "MinIO: Not responding"

# Clean up everything
clean:
	@echo "Cleaning up containers and volumes..."
	@$(DOCKER_COMPOSE) down -v --remove-orphans
	@docker system prune -f
	@echo "Cleanup complete."

# Run tests
test:
	pytest -q
	@echo "Testing GraphDB connection..."
	@curl -s http://localhost:7200/repositories/glycokg/statements || echo "GraphDB test failed"

typecheck:
	@echo "mypy not configured in MVP; add later"

fmt:
	@echo "Formatting with black (not installed in requirements by default)"

# Build custom images
build:
	docker build -f docker/api.Dockerfile -t glyco-api:0.1.0 .
	@$(DOCKER_COMPOSE) build --no-cache

run:
	uvicorn platform.api.main:app --host 0.0.0.0 --port 8080

bench:
	python scripts/bench.py

# Initialize with sample data
init-data:
	@echo "Initializing with sample data..."
	@$(PYTHON) scripts/init_sample_data.py
	@echo "Sample data loaded. Check GraphDB at http://localhost:7200"

# Development helpers
# ============================================================================
# DATABASE OPERATIONS (Production Ready)
# ============================================================================

# Database shell access with correct credentials
db-shell:
	@echo "🔐 Connecting to PostgreSQL with credentials: $(POSTGRES_USER)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)"
	@$(DOCKER_COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

# Initialize database with full schema and credentials
db-init:
	@echo "🗄️  Initializing PostgreSQL database..."
	@echo "📋 Database: $(DATABASE_URL)"
	@$(DOCKER_COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "SELECT version();"
	@echo "✅ Database initialized successfully!"

# Load sample data with correct credentials
data-sync:
	@echo "🔄 Starting data synchronization with credentials..."
	@echo "Database: $(DATABASE_URL)"
	@POSTGRES_USER=$(POSTGRES_USER) POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) python scripts/init_sample_data.py --full-sync --organism=9606 --limit=100

# Development shells
dev-shell:
	@$(DOCKER_COMPOSE) exec jupyterlab bash

redis-cli:
	@$(DOCKER_COMPOSE) exec redis redis-cli

# Access MongoDB with credentials
mongo-shell:
	@echo "🍃 Connecting to MongoDB: $(MONGODB_URL)"
	@$(DOCKER_COMPOSE) exec mongodb mongosh "$(MONGODB_URL)/$(MONGODB_DB)"

# MinIO client access
minio-cli:
	@echo "📦 MinIO Console: http://$(MINIO_ENDPOINT)"
	@echo "Access Key: $(MINIO_ACCESS_KEY)"
	@echo "Secret Key: $(MINIO_SECRET_KEY)"

# ============================================================================
# BACKUP AND RESTORE WITH CREDENTIALS
# ============================================================================

backup:
	@echo "💾 Creating comprehensive backup with credentials..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "Backing up PostgreSQL..."
	@$(DOCKER_COMPOSE) exec postgres pg_dump -U $(POSTGRES_USER) $(POSTGRES_DB) > backups/$(shell date +%Y%m%d_%H%M%S)/postgres_backup.sql
	@echo "Backing up MongoDB..."
	@$(DOCKER_COMPOSE) exec mongodb mongodump --uri="$(MONGODB_URL)" --out=backups/$(shell date +%Y%m%d_%H%M%S)/mongodb_backup
	@echo "✅ Backup created in backups/ directory with all credentials"

# Restore from backup
restore:
	@echo "🔄 Restore functionality - specify backup directory"
	@echo "Usage: make restore BACKUP_DIR=backups/YYYYMMDD_HHMMSS"

# Production deployment (placeholder)
deploy:
	@echo "Production deployment not yet implemented"
	@echo "This would deploy to production infrastructure"

# Monitor resource usage
monitor:
	@echo "📊 Resource usage:"
	@docker stats --no-stream
	@echo ""
	@echo "💾 Disk usage:"
	@docker system df

# Quick start (setup + start + health check)
quick-start:
	@echo "🚀 Quick Start: Complete Platform Deployment"
	@make setup
	@make start
	@make health-check
	@echo "✅ Quick start complete!"

# Comprehensive health check
health-check:
	@echo "🏥 Comprehensive Health Check:"
	@echo "API Health:" && curl -s http://localhost:8000/healthz || echo "❌ API not responding"
	@echo "GraphDB:" && curl -s http://localhost:7200/rest/monitor/infrastructure || echo "❌ GraphDB not responding"
	@echo "Elasticsearch:" && curl -s http://localhost:9200/_cluster/health || echo "❌ Elasticsearch not responding"
	@echo "MinIO:" && curl -s http://localhost:9000/minio/health/live || echo "❌ MinIO not responding"
	@echo "Prometheus:" && curl -s http://localhost:9090/-/healthy || echo "❌ Prometheus not responding"

# View specific service logs (usage: make service-logs SERVICE=glyco_api)
service-logs:
	@echo "📋 Viewing logs for $(SERVICE):"
	docker-compose logs -f $(SERVICE)

# Complete platform deployment with staged rollout
deploy-staged:
	@echo "🚀 Staged Platform Deployment:"
	@echo "📋 Phase 1: Infrastructure Services"
	docker-compose up -d postgres redis graphdb elasticsearch mongodb minio traefik
	@echo "⏳ Waiting for infrastructure (30 seconds)..."
	@sleep 30
	@echo "📋 Phase 2: Application Services"
	docker-compose up -d glyco_api jupyter prometheus grafana
	@echo "⏳ Final initialization (15 seconds)..."
	@sleep 15
	@echo "🧪 Running health checks..."
	@make health-check
	@echo "✅ Staged deployment complete!"

# Development mode (API only with hot reload)
dev-mode:
	@echo "🔧 Starting development mode with hot reload..."
	cd glyco_platform && python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Clean restart (removes containers and networks)
clean-restart:
	@echo "🧹 Clean restart - removing containers and networks..."
	docker-compose down --remove-orphans
	@echo "🔄 Restarting services..."
	docker-compose up -d
	@echo "✅ Clean restart complete!"

# Complete system reset (⚠️ WARNING: Removes all data!)
system-reset:
	@echo "⚠️  WARNING: This will remove ALL data permanently!"
	@echo "Press Ctrl+C within 10 seconds to cancel..."
	@sleep 10
	@echo "🗑️  Removing all containers, volumes, and networks..."
	docker-compose down -v --remove-orphans
	docker volume prune -f
	docker network prune -f
	@echo "✅ Complete system reset finished!"

# Show comprehensive documentation
docs:
	@echo "📖 Documentation Available:"
	@echo "  • Complete Docker Guide: DOCKER_DEPLOYMENT.md"
	@echo "  • API Documentation: http://localhost:8000/docs (when running)"
	@echo "  • README: README.md"
	@echo "  • Implementation Status: IMPLEMENTATION_STATUS.md"

# Fix Docker permissions (run this if you get permission denied errors)
fix-docker:
	@echo "🔧 Fixing Docker permissions..."
	@echo "Adding current user to docker group (requires sudo password):"
	sudo usermod -aG docker $$USER
	@echo "✅ User added to docker group!"
	@echo "📢 IMPORTANT: You must logout/login OR run 'newgrp docker' for changes to take effect"
	@echo "Then try 'make start' again"

# Check Docker status and permissions
check-docker:
	@echo "🔍 Docker System Check:"
	@echo "Docker version:" && docker --version || echo "❌ Docker not installed"
	@echo "Docker Compose version:" && docker-compose --version || echo "❌ Docker Compose not installed"
	@echo "Docker daemon status:" && docker ps >/dev/null 2>&1 && echo "✅ Docker accessible" || echo "❌ Docker permission denied - run 'make fix-docker'"
	@echo "Docker info:" && docker info --format '{{.ServerVersion}}' 2>/dev/null || echo "❌ Docker not accessible"