# 🚀 Complete Deployment Guide - Glycoinformatics AI Platform

**Production Ready Deployment Instructions**  
**Version:** 0.1.0  
**Date:** November 2, 2025  
**System User:** Adebayo  

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [System Credentials](#system-credentials)
- [Database Configuration](#database-configuration)
- [Quick Start Deployment](#quick-start-deployment)
- [Service Access URLs](#service-access-urls)
- [Data Pipeline Commands](#data-pipeline-commands)
- [Troubleshooting](#troubleshooting)
- [Production Checklist](#production-checklist)

## 🔧 Prerequisites

### System Requirements
- **Operating System:** Ubuntu 20.04+ / Linux
- **Docker:** 20.10+
- **Docker Compose:** 1.29+
- **Memory:** 8GB+ RAM
- **Storage:** 50GB+ free space
- **Network:** Internet connection for external APIs

### User Setup
```bash
# Add user to docker group (use system password: Adebayo@120)
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker access
docker --version
docker-compose --version
```

## 🔐 System Credentials

### Primary System Password
```
System User Password: Adebayo@120
```

### Database Credentials (Production Ready)

#### PostgreSQL Primary Database
```
Host: localhost
Port: 5432
Database: glycokg
Username: glyco_admin
Password: glyco_secure_pass_2025
Connection URL: postgresql://glyco_admin:glyco_secure_pass_2025@localhost:5432/glycokg
```

#### Redis Cache Database
```
Host: localhost
Port: 6379
Database: 0
URL: redis://localhost:6379/0
```

#### MongoDB Document Database
```
Host: localhost
Port: 27017
Username: glyco_admin
Password: glyco_secure_pass_2025
Database: glyco_results
URL: mongodb://glyco_admin:glyco_secure_pass_2025@localhost:27017
```

#### MinIO Object Storage
```
Endpoint: localhost:9000
Access Key: glyco_admin
Secret Key: glyco_secure_pass_2025
Bucket: glyco-data
Console: http://localhost:9001
```

#### GraphDB RDF Knowledge Graph
```
URL: http://localhost:7200
Repository: glycokg
Username: (none - open access)
Password: (none - open access)
```

#### Elasticsearch Search Engine
```
URL: http://localhost:9200
Username: (none - security disabled)
Password: (none - security disabled)
```

## 🚀 Quick Start Deployment

### One-Command Full Deployment

```bash
# Clone repository
git clone https://github.com/adetayookunoye/GLYCOINFORMATICS_AI.git
cd GLYCOINFORMATICS_AI

# Complete deployment with credentials
make docker-start

# Initialize database with sample data
make data-sync

# Verify all services
make status
```

### Step-by-Step Deployment

```bash
# 1. Verify system prerequisites
make setup

# 2. Start all Docker services (11 services)
sudo docker-compose up -d

# 3. Verify services are running
docker-compose ps

# 4. Initialize databases with credentials
make db-init

# 5. Load sample glycoinformatics data
POSTGRES_USER=glyco_admin POSTGRES_PASSWORD=glyco_secure_pass_2025 python scripts/init_sample_data.py --full-sync --organism=9606 --limit=100

# 6. Verify data loaded successfully
make db-shell
# In PostgreSQL shell:
\dt cache.*
SELECT COUNT(*) FROM cache.glycan_structures;
\q
```

## 🌐 Service Access URLs

After successful deployment, access your services:

### Core Platform Services
- **API Documentation:** http://localhost:8000/docs
- **API Health Check:** http://localhost:8000/healthz
- **GraphQL Interface:** http://localhost:8000/graphql
- **SPARQL Endpoint:** http://localhost:8000/sparql

### Database Management
- **GraphDB Workbench:** http://localhost:7200
- **MinIO Console:** http://localhost:9001
  - Username: `glyco_admin`
  - Password: `glyco_secure_pass_2025`

### Development Tools
- **Jupyter Lab:** http://localhost:8888
  - Token: `glycoinfo`
- **Traefik Dashboard:** http://localhost:8080

### Monitoring (Future Implementation)
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000

## 💾 Data Pipeline Commands

### Sample Data Initialization
```bash
# Load sample data with all credentials
make data-sync

# Alternative: Direct command with credentials
POSTGRES_USER=glyco_admin POSTGRES_PASSWORD=glyco_secure_pass_2025 \
python scripts/init_sample_data.py --full-sync --organism=9606 --limit=100
```

### Database Operations
```bash
# PostgreSQL shell access
make db-shell

# MongoDB shell access  
make mongo-shell

# Redis CLI access
make redis-cli

# Create backup of all databases
make backup
```

### Data Verification
```bash
# Check PostgreSQL data
make db-shell
SELECT COUNT(*) FROM cache.glycan_structures;
SELECT COUNT(*) FROM cache.protein_glycan_associations;
\q

# Check GraphDB data
curl -X POST http://localhost:7200/repositories/glycokg/statements \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Docker Permission Issues
```bash
# Error: Permission denied accessing Docker socket
sudo usermod -aG docker $USER
newgrp docker
# Or restart terminal session
```

#### Database Connection Issues
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Reset database
docker-compose restart postgres
make db-init
```

#### Port Conflicts
```bash
# Check what's using port 5432
sudo netstat -tlnp | grep 5432

# Stop conflicting services
sudo systemctl stop postgresql
```

#### Memory Issues
```bash
# Check Docker memory usage
docker stats

# Increase Docker memory limits
# Edit /etc/docker/daemon.json:
{
  "default-runtime": "runc",
  "default-shm-size": "1g"
}
```

### Service-Specific Troubleshooting

#### GraphDB Issues
```bash
# Check GraphDB status
curl -f http://localhost:7200/rest/repositories

# View GraphDB logs
docker-compose logs graphdb

# Reset GraphDB data
docker-compose down
docker volume rm glycoinformatics_ai_graphdb_data
docker-compose up -d graphdb
```

#### API Issues
```bash
# Check FastAPI health
curl http://localhost:8000/healthz

# View API logs
docker-compose logs glyco-api

# Restart API service
docker-compose restart glyco-api
```

## ✅ Production Checklist

### Pre-Deployment Verification
- [ ] Docker and Docker Compose installed
- [ ] User added to docker group
- [ ] Minimum 8GB RAM available
- [ ] Minimum 50GB disk space available
- [ ] Internet connection available
- [ ] Ports 5432, 6379, 7200, 8000, 9000, 9200, 27017 available

### Post-Deployment Verification
- [ ] All 9 Docker containers running: `docker-compose ps`
- [ ] PostgreSQL accepting connections: `make db-shell`
- [ ] GraphDB accessible: `curl http://localhost:7200`
- [ ] API responding: `curl http://localhost:8000/healthz`
- [ ] Sample data loaded: `make data-sync`
- [ ] MinIO console accessible: http://localhost:9001

### Security Checklist
- [ ] Database passwords changed from defaults
- [ ] MinIO access keys rotated
- [ ] Firewall configured for production
- [ ] SSL certificates installed (production)
- [ ] Backup strategy implemented

### Performance Checklist
- [ ] Database indexes created
- [ ] Redis memory limits configured
- [ ] Elasticsearch heap size optimized
- [ ] Docker resource limits set
- [ ] Monitoring configured

## 🔄 Maintenance Commands

### Daily Operations
```bash
# Check system health
make status

# View logs
make logs

# Create backup
make backup
```

### Weekly Operations
```bash
# Clean up old data
make clean

# Update containers
docker-compose pull
docker-compose up -d
```

### Monthly Operations
```bash
# Full system backup
make backup

# Update dependencies
pip install -r requirements.txt --upgrade
```

## 📞 Support Information

### Documentation References
- **Complete Documentation:** `/documentations/`
- **API Documentation:** http://localhost:8000/docs
- **Architecture Guide:** `/documentations/COMPREHENSIVE_DATA_ARCHITECTURE.md`
- **Implementation Status:** `/documentations/IMPLEMENTATION_STATUS.md`

### Repository Information
- **GitHub Repository:** https://github.com/adetayookunoye/GLYCOINFORMATICS_AI
- **Issues:** https://github.com/adetayookunoye/GLYCOINFORMATICS_AI/issues
- **Documentation:** https://github.com/adetayookunoye/GLYCOINFORMATICS_AI/tree/main/documentations

### Emergency Contacts
- **System Administrator:** Adebayo
- **Platform Version:** v0.1.0
- **Deployment Date:** November 2, 2025

---

**🚀 Your Glycoinformatics AI Platform is ready for production deployment!**

*This guide contains all necessary credentials and step-by-step instructions for complete platform deployment. Keep this document secure and update credentials as needed for production environments.*