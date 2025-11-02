# 🐳 Complete Docker Deployment Guide
## Glycoinformatics AI Platform

### 📋 Overview
This guide provides **complete instructions** for deploying the Glycoinformatics AI Platform using Docker. The platform includes 11 production-ready services providing a comprehensive glycoinformatics research environment.

---

## 🚀 Quick Start (One Command)

### Prerequisites Check
```bash
# 1. Verify Docker is installed
docker --version
docker-compose --version

# 2. Check Docker access (should not require sudo)
docker ps

# If docker ps fails, run:
sudo usermod -aG docker $USER
newgrp docker  # or logout/login
```

### Complete Deployment
```bash
# Navigate to project directory
cd glycoinformatics_ai

# Option 1: Use our enhanced setup script
./setup_foundation.sh start-all

# Option 2: Use Docker Compose directly
docker-compose up -d
```

---

## 🏗️ Detailed Deployment Instructions

### Step 1: System Prerequisites

#### Install Docker
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# Or install Docker Desktop from:
# https://docs.docker.com/get-docker/
```

#### Configure Docker Access
```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Apply group membership (choose one):
newgrp docker           # Option 1: Apply in current session
# OR logout and login     # Option 2: Apply system-wide
```

#### Verify Docker Setup
```bash
docker --version        # Should show Docker version
docker-compose --version # Should show Compose version
docker ps              # Should work without sudo
```

### Step 2: Project Setup

#### Clone/Navigate to Project
```bash
cd glycoinformatics_ai
```

#### Create Required Directories
```bash
mkdir -p data/{raw,interim,processed}
mkdir -p logs models
mkdir -p infrastructure/{prometheus,grafana,postgres/init,redis,graphdb}
```

#### Create Environment File (Optional)
```bash
# Create .env file for custom configuration
cat > .env << 'EOF'
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
ENVIRONMENT=production
DEBUG=false

# External Service URLs
GRAPHDB_URL=http://graphdb:7200
ELASTICSEARCH_URL=http://elasticsearch:9200
MONGODB_URL=mongodb://glyco_admin:glyco_secure_pass_2025@mongodb:27017

# MinIO Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=glyco_admin
MINIO_SECRET_KEY=glyco_secure_pass_2025
EOF
```

### Step 3: Deploy Complete Platform

#### Method 1: Enhanced Setup Script (Recommended)
```bash
# Make setup script executable
chmod +x setup_foundation.sh

# Deploy complete platform
./setup_foundation.sh start-all
```

#### Method 2: Docker Compose Direct
```bash
# Start all services
docker-compose up -d

# View startup logs
docker-compose logs -f
```

#### Method 3: Staged Deployment
```bash
# Phase 1: Infrastructure services
docker-compose up -d postgres redis graphdb elasticsearch mongodb minio traefik

# Wait for services to initialize
sleep 30

# Phase 2: Application services  
docker-compose up -d glyco_api jupyter prometheus grafana

# Final verification
docker-compose ps
```

---

## 🌐 Service Access Information

### Production Services (11 Total)

After deployment, access your services at:

| Service | URL | Purpose | Credentials |
|---------|-----|---------|-------------|
| **FastAPI Application** | http://localhost:8000 | Main API | - |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs | - |
| **API Health Check** | http://localhost:8000/healthz | Service health | - |
| **GraphDB Workbench** | http://localhost:7200 | RDF data management | - |
| **Elasticsearch** | http://localhost:9200 | Search engine | - |
| **MinIO Console** | http://localhost:9001 | Object storage UI | glyco_admin / glyco_secure_pass_2025 |
| **Jupyter Lab** | http://localhost:8888 | Development environment | - |
| **Traefik Dashboard** | http://localhost:8080 | Load balancer status | - |
| **Prometheus** | http://localhost:9090 | Metrics collection | - |
| **Grafana** | http://localhost:3000 | Monitoring dashboards | admin / admin |

### Database Connections

| Database | Host | Port | Database | Username | Password |
|----------|------|------|----------|----------|----------|
| **PostgreSQL** | localhost | 5432 | glycokg | glyco_admin | glyco_secure_pass_2025 |
| **MongoDB** | localhost | 27017 | - | glyco_admin | glyco_secure_pass_2025 |
| **Redis** | localhost | 6379 | 0 | - | - |

---

## 🔧 Management Commands

### Service Management
```bash
# Check service status
docker-compose ps

# View all logs
docker-compose logs

# View specific service logs  
docker-compose logs glyco_api
docker-compose logs postgres

# Stop all services
docker-compose down

# Restart all services
docker-compose restart

# Start specific service
docker-compose up -d glyco_api
```

### Health Checks
```bash
# API health check
curl http://localhost:8000/healthz

# Database connectivity
docker-compose exec postgres pg_isready -U glyco_admin

# Redis connectivity  
docker-compose exec redis redis-cli ping

# Service resource usage
docker stats
```

### Data Management
```bash
# Database backup
docker-compose exec postgres pg_dump -U glyco_admin glycokg > backup.sql

# View database
docker-compose exec postgres psql -U glyco_admin -d glycokg

# Access MongoDB
docker-compose exec mongodb mongo -u glyco_admin -p glyco_secure_pass_2025

# MinIO file management (via web console at localhost:9001)
```

---

## 🧹 Maintenance Operations

### Clean Restart
```bash
# Stop services
docker-compose down

# Remove containers and networks  
docker-compose down --remove-orphans

# Clean system (removes unused containers/images)
docker system prune -f

# Restart services
docker-compose up -d
```

### Complete Reset (⚠️ Removes all data!)
```bash
# WARNING: This removes all data permanently
docker-compose down -v --remove-orphans
docker volume prune -f
docker network prune -f
docker system prune -a -f

# Then restart
docker-compose up -d
```

### Update Images
```bash
# Pull latest images
docker-compose pull

# Rebuild custom images
docker-compose build --no-cache

# Restart with new images
docker-compose up -d
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Docker Permission Denied
```bash
# Problem: "permission denied while trying to connect to Docker daemon"
# Solution:
sudo usermod -aG docker $USER
newgrp docker
# OR logout and login
```

#### Port Already in Use
```bash
# Problem: "Port is already allocated"
# Solution: Check what's using the port
sudo netstat -tulpn | grep :8000

# Kill conflicting process
sudo kill <PID>

# Or change port in docker-compose.yml
```

#### Services Not Starting
```bash
# Check logs for errors
docker-compose logs <service_name>

# Check system resources
docker stats
df -h
free -h

# Restart specific service
docker-compose restart <service_name>
```

#### Database Connection Issues
```bash
# Check database is running
docker-compose ps postgres

# Test database connectivity
docker-compose exec postgres pg_isready -U glyco_admin

# Reset database
docker-compose restart postgres
```

#### Out of Disk Space
```bash
# Clean Docker system
docker system prune -a -f --volumes

# Remove unused images
docker image prune -a -f

# Check disk usage
docker system df
```

### Service-Specific Troubleshooting

#### API Service Issues
```bash
# Check API logs
docker-compose logs glyco_api

# Test API directly
curl http://localhost:8000/healthz

# Restart API
docker-compose restart glyco_api
```

#### Database Issues
```bash
# PostgreSQL logs
docker-compose logs postgres

# Access PostgreSQL console
docker-compose exec postgres psql -U glyco_admin -d glycokg

# MongoDB logs  
docker-compose logs mongodb

# Redis logs
docker-compose logs redis
```

#### Resource Monitoring
```bash
# Real-time container stats
docker stats

# Check service resource usage
docker-compose exec glyco_api htop

# System resources
df -h
free -h
```

---

## 📊 Monitoring and Metrics

### Built-in Monitoring
- **Prometheus**: Metrics collection at http://localhost:9090
- **Grafana**: Dashboards and alerts at http://localhost:3000
- **API Metrics**: Available at http://localhost:8000/metrics

### Custom Monitoring
```bash
# View container metrics
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# API response time monitoring
while true; do curl -o /dev/null -s -w "%{time_total}\n" http://localhost:8000/healthz; sleep 1; done

# Database monitoring
docker-compose exec postgres psql -U glyco_admin -d glycokg -c "SELECT * FROM pg_stat_activity;"
```

---

## 🚀 Performance Optimization

### Resource Allocation
Edit `docker-compose.yml` to adjust resource limits:

```yaml
services:
  glyco_api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"
```

### Database Optimization
```bash
# Optimize PostgreSQL
docker-compose exec postgres psql -U glyco_admin -d glycokg -c "VACUUM ANALYZE;"

# MongoDB indexing
docker-compose exec mongodb mongo -u glyco_admin -p glyco_secure_pass_2025 --eval "db.collection.ensureIndex({field: 1})"
```

---

## 🔒 Security Considerations

### Production Deployment
1. **Change default passwords** in docker-compose.yml
2. **Use environment files** for sensitive data
3. **Enable TLS/SSL** for external access
4. **Configure firewall** to restrict access
5. **Regular security updates** for all components

### Secure Configuration
```bash
# Create secure environment file
cat > .env.production << 'EOF'
POSTGRES_PASSWORD=$(openssl rand -base64 32)
MONGODB_PASSWORD=$(openssl rand -base64 32)
MINIO_SECRET_KEY=$(openssl rand -base64 32)
EOF

# Set restrictive permissions
chmod 600 .env.production
```

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] All 11 services are running: `docker-compose ps`
- [ ] API responds: `curl http://localhost:8000/healthz`
- [ ] API docs accessible: http://localhost:8000/docs
- [ ] GraphDB workbench loads: http://localhost:7200
- [ ] Grafana dashboards load: http://localhost:3000
- [ ] No error logs: `docker-compose logs | grep -i error`
- [ ] Adequate disk space: `df -h`
- [ ] Reasonable resource usage: `docker stats`

---

## 📚 Additional Resources

### Documentation
- **API Documentation**: Available at http://localhost:8000/docs after deployment
- **Project README**: Complete platform overview and architecture
- **Component READMEs**: Detailed documentation in each service directory

### Support
- **Logs**: Check `docker-compose logs` for detailed error information
- **Health Checks**: Use `curl http://localhost:8000/healthz` for API status
- **System Monitoring**: Access Grafana dashboards at http://localhost:3000

### Development
- **Local Development**: Use `./setup_foundation.sh dev` for development mode
- **Testing**: Run test suite with proper environment setup
- **Hot Reload**: API supports hot reload in development mode

---

## 🎉 Success!

Your **Glycoinformatics AI Platform** is now deployed with:

✅ **11 Production Services** running in containers  
✅ **Complete API** with interactive documentation  
✅ **Monitoring Stack** with Prometheus and Grafana  
✅ **Database Cluster** with PostgreSQL, MongoDB, Redis  
✅ **Development Environment** with Jupyter Lab  
✅ **Load Balancing** with Traefik  
✅ **Object Storage** with MinIO  
✅ **Search Engine** with Elasticsearch  
✅ **Knowledge Graph** with GraphDB  

**Start exploring**: http://localhost:8000/docs

Happy researching! 🧬✨