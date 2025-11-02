#!/bin/bash

# Glycoinformatics AI Platform - Foundation Startup Script
# This script helps you get started with the core platform files

set -e

echo "🧬 Glycoinformatics AI Platform - Foundation Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check Docker permissions
if ! docker ps &> /dev/null; then
    print_error "Docker permission denied. Your user needs to be in the docker group."
    echo
    print_info "To fix this, run the following commands:"
    echo "  sudo usermod -aG docker \$USER"
    echo "  newgrp docker  # or log out and back in"
    echo
    print_info "Then try running this script again."
    exit 1
fi

print_status "Docker and Docker Compose are installed and accessible"

# Create necessary directories
echo
print_info "Creating necessary directories..."

directories=(
    "data/raw"
    "data/interim" 
    "data/processed"
    "logs"
    "models"
    "infrastructure/prometheus"
    "infrastructure/grafana"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    else
        print_warning "Directory already exists: $dir"
    fi
done

# Create basic configuration files if they don't exist
echo
print_info "Setting up basic configuration files..."

# Create basic prometheus config if it doesn't exist
if [ ! -f "infrastructure/prometheus/prometheus.yml" ]; then
    cat > infrastructure/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'glyco-api'
    static_configs:
      - targets: ['glyco-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    print_status "Created prometheus configuration"
fi

# Create environment file for development
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Glycoinformatics AI Platform Environment Configuration

# Database Configuration
POSTGRES_DB=glycokg
POSTGRES_USER=glyco_admin
POSTGRES_PASSWORD=glyco_secure_pass_2025

# Redis Configuration  
REDIS_URL=redis://redis:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Development Settings
ENVIRONMENT=development
DEBUG=true

# External Service URLs
GRAPHDB_URL=http://graphdb:7200
ELASTICSEARCH_URL=http://elasticsearch:9200
MONGODB_URL=mongodb://glyco_admin:glyco_secure_pass_2025@mongodb:27017

# MinIO Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=glyco_admin
MINIO_SECRET_KEY=glyco_secure_pass_2025
EOF
    print_status "Created environment configuration file"
fi

# Function to start the infrastructure
start_infrastructure() {
    echo
    print_info "Starting infrastructure services..."
    
    # Start infrastructure services first (without API)
    docker-compose up -d postgres redis graphdb elasticsearch minio mongodb traefik
    
    print_status "Infrastructure services started"
    
    # Wait for services to be ready
    echo
    print_info "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    echo "Waiting for PostgreSQL..."
    until docker-compose exec -T postgres pg_isready -U glyco_admin -d glycokg; do
        sleep 2
    done
    print_status "PostgreSQL is ready"
    
    # Wait for Redis
    echo "Waiting for Redis..."
    until docker-compose exec -T redis redis-cli ping; do
        sleep 2
    done
    print_status "Redis is ready"
    
    # Wait for GraphDB
    echo "Waiting for GraphDB..."
    until curl -s http://localhost:7200/rest/repositories > /dev/null; do
        sleep 5
    done
    print_status "GraphDB is ready"
    
    # Wait for Elasticsearch
    echo "Waiting for Elasticsearch..."
    until curl -s http://localhost:9200/_cluster/health > /dev/null; do
        sleep 5
    done
    print_status "Elasticsearch is ready"
    
    print_status "All infrastructure services are ready!"
}

# Function to build and start API
start_api() {
    echo
    print_info "Building and starting API service..."
    
    # Build the API Docker image
    docker-compose build glyco_api
    print_status "API image built"
    
    # Start the API service
    docker-compose up -d glyco_api
    print_status "API service started"
    
    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    until curl -s http://localhost:8000/healthz > /dev/null; do
        sleep 2
    done
    print_status "API is ready and responding!"
}

# Function to run tests
run_tests() {
    echo
    print_info "Running basic tests..."
    
    # Install dependencies if not in Docker
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        print_status "Virtual environment created and dependencies installed"
    else
        source venv/bin/activate
    fi
    
    # Run tests
    python -m pytest tests/test_api.py -v
    print_status "Tests completed"
}

# Function to show service status
show_status() {
    echo
    print_info "Service Status:"
    docker-compose ps
    
    echo
    print_info "Service URLs:"
    echo "🌐 API Documentation: http://localhost:8000/docs"
    echo "📊 GraphDB Workbench: http://localhost:7200"
    echo "🔍 Elasticsearch: http://localhost:9200"
    echo "📁 MinIO Console: http://localhost:9001"
    echo "🧪 Jupyter Lab: http://localhost:8888"
    echo "📈 Traefik Dashboard: http://localhost:8080"
    echo "💾 API Health: http://localhost:8000/healthz"
    echo "📋 API Metrics: http://localhost:8000/metrics"
}

# Main script logic
case "${1:-help}" in
    "start-infra")
        start_infrastructure
        show_status
        ;;
    "start-api")
        start_api
        show_status
        ;;
    "start-all")
        start_infrastructure
        start_api
        show_status
        ;;
    "test")
        run_tests
        ;;
    "status")
        show_status
        ;;
    "stop")
        print_info "Stopping all services..."
        docker-compose down
        print_status "All services stopped"
        ;;
    "logs")
        service=${2:-glyco_api}
        print_info "Showing logs for $service..."
        docker-compose logs -f $service
        ;;
    "help"|*)
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  start-infra    Start infrastructure services only (databases, etc.)"
        echo "  start-api      Start the API service (requires infrastructure)"
        echo "  start-all      Start all services"
        echo "  test           Run the test suite"
        echo "  status         Show service status and URLs"
        echo "  stop           Stop all services"
        echo "  logs [service] Show logs for a service (default: glyco_api)"
        echo "  help           Show this help message"
        echo
        echo "Examples:"
        echo "  $0 start-all     # Start everything"
        echo "  $0 logs postgres # Show PostgreSQL logs"
        echo "  $0 test          # Run tests"
        echo
        ;;
esac

echo
echo "🧬 Foundation setup complete! Your Glycoinformatics AI Platform is ready."