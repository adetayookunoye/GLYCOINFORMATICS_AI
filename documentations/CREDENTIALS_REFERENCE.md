# 🔐 Credentials Reference - Glycoinformatics AI Platform

**CONFIDENTIAL - Production Credentials**  
**Version:** 0.1.0  
**Date:** November 2, 2025  
**System Owner:** Adebayo  

## ⚠️ Security Notice

This document contains production credentials and passwords. Keep this information secure and limit access to authorized personnel only.

## 🖥️ System Access

### Primary System User
```
Username: adetayo
Password: Adebayo@120
Home Directory: /home/adetayo
Platform Directory: /home/adetayo/Documents/CSCI Forms/Adetayo Research/glycoinformatics_ai_v0.1.0/glycoinformatics_ai
```

### Docker Access
```bash
# Add user to docker group (requires system password)
sudo usermod -aG docker $USER
# Password: Adebayo@120

# Access without sudo
newgrp docker
```

## 🗄️ Database Credentials

### PostgreSQL Primary Database
```
Connection String: postgresql://glyco_admin:glyco_secure_pass_2025@localhost:5432/glycokg

Components:
- Host: localhost
- Port: 5432
- Database: glycokg
- Username: glyco_admin
- Password: glyco_secure_pass_2025

Docker Command: 
docker-compose exec postgres psql -U glyco_admin -d glycokg

Makefile Command:
make db-shell
```

### Redis Cache Database
```
Connection String: redis://localhost:6379/0

Components:
- Host: localhost
- Port: 6379
- Database: 0
- Password: (none - no auth required)

Docker Command:
docker-compose exec redis redis-cli

Makefile Command:
make redis-cli
```

### MongoDB Document Database
```
Connection String: mongodb://glyco_admin:glyco_secure_pass_2025@localhost:27017

Components:
- Host: localhost
- Port: 27017
- Username: glyco_admin
- Password: glyco_secure_pass_2025
- Database: glyco_results

Docker Command:
docker-compose exec mongodb mongosh "mongodb://glyco_admin:glyco_secure_pass_2025@localhost:27017/glyco_results"

Makefile Command:
make mongo-shell
```

### MinIO Object Storage
```
Console URL: http://localhost:9001
API Endpoint: localhost:9000

Credentials:
- Access Key: glyco_admin
- Secret Key: glyco_secure_pass_2025
- Default Bucket: glyco-data

Docker Command:
docker-compose exec minio mc alias set local http://localhost:9000 glyco_admin glyco_secure_pass_2025

Makefile Command:
make minio-cli
```

### GraphDB RDF Knowledge Graph
```
Workbench URL: http://localhost:7200
Repository: glycokg

Credentials:
- Username: (none - open access)
- Password: (none - open access)

SPARQL Endpoint: http://localhost:7200/repositories/glycokg
```

### Elasticsearch Search Engine
```
URL: http://localhost:9200
Cluster Name: glyco-cluster

Credentials:
- Username: (none - security disabled)
- Password: (none - security disabled)

Health Check: http://localhost:9200/_cluster/health
```

## 🚀 Application Access

### FastAPI Application
```
API Base URL: http://localhost:8000
Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/healthz

Internal Docker URL: http://glyco-api:8000
```

### Jupyter Lab Development
```
URL: http://localhost:8888
Token: glycoinfo

Notebook Directory: /home/jovyan/work (mapped to project root)
```

### Traefik Load Balancer
```
Dashboard URL: http://localhost:8080
Configuration: Auto-discovery via Docker labels
```

## 🔄 Environment Variables

### Production .env File
```bash
# Database URLs
DATABASE_URL=postgresql://glyco_admin:glyco_secure_pass_2025@localhost:5432/glycokg
REDIS_URL=redis://localhost:6379/0
MONGODB_URL=mongodb://glyco_admin:glyco_secure_pass_2025@localhost:27017
GRAPHDB_URL=http://localhost:7200
ELASTICSEARCH_URL=http://localhost:9200

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=glyco_admin
MINIO_SECRET_KEY=glyco_secure_pass_2025

# System Password
SYSTEM_USER_PASSWORD=Adebayo@120
```

### Data Pipeline Environment
```bash
# For data synchronization scripts
export POSTGRES_USER=glyco_admin
export POSTGRES_PASSWORD=glyco_secure_pass_2025
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=glycokg

# Run data sync with credentials
POSTGRES_USER=glyco_admin POSTGRES_PASSWORD=glyco_secure_pass_2025 python scripts/init_sample_data.py --full-sync
```

## 🔧 Makefile Integration

### Database Commands with Credentials
```bash
# All database operations use embedded credentials
make db-shell      # PostgreSQL access
make mongo-shell   # MongoDB access  
make redis-cli     # Redis access
make data-sync     # Data synchronization
make backup        # Full backup with credentials
```

### Service Management
```bash
make start         # Start all services
make stop          # Stop all services
make restart       # Restart all services
make status        # Check service health
make logs          # View all logs
```

## 🛡️ Security Best Practices

### Password Management
- [ ] Credentials stored in .env file (not in source code)
- [ ] .env file added to .gitignore
- [ ] Production passwords different from development
- [ ] Regular password rotation schedule implemented

### Access Control
- [ ] Database access restricted to application network
- [ ] MinIO console access secured
- [ ] GraphDB workbench secured (future implementation)
- [ ] API authentication implemented (future enhancement)

### Backup Security
- [ ] Backup files encrypted
- [ ] Backup storage secured
- [ ] Backup restore procedures tested
- [ ] Backup retention policy defined

## 📋 Quick Reference Commands

### Essential Operations
```bash
# Start platform with all credentials
sudo docker-compose up -d
make data-sync

# Check all services
make status

# Access databases
make db-shell     # PostgreSQL
make mongo-shell  # MongoDB  
make redis-cli    # Redis

# Create backup
make backup

# View logs
make logs

# Stop everything
sudo docker-compose down
```

### Troubleshooting Commands
```bash
# Check Docker status
docker --version
docker-compose --version
docker-compose ps

# Check database connections
docker-compose exec postgres psql -U glyco_admin -d glycokg -c "SELECT version();"
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
docker-compose exec redis redis-cli ping

# Check API health
curl http://localhost:8000/healthz
```

## 📞 Emergency Access

### If Docker Issues
```bash
# Reset Docker daemon (requires system password: Adebayo@120)
sudo systemctl restart docker
sudo docker-compose up -d
```

### If Database Corruption
```bash
# PostgreSQL recovery
sudo docker-compose down
sudo docker volume rm glycoinformatics_ai_postgres_data
sudo docker-compose up -d postgres
make db-init
make data-sync
```

### If Complete Reset Needed
```bash
# Full platform reset (requires system password: Adebayo@120)  
sudo docker-compose down -v
sudo docker system prune -f
sudo docker-compose up -d
make data-sync
```

## 🔄 Credential Rotation Schedule

### Monthly Rotation (Recommended)
- [ ] MinIO access keys
- [ ] MongoDB passwords
- [ ] PostgreSQL passwords

### Quarterly Rotation (Required)
- [ ] System user password
- [ ] Application secrets
- [ ] API tokens (when implemented)

### Annual Rotation (Mandatory)
- [ ] All service passwords
- [ ] SSL certificates
- [ ] Backup encryption keys

---

**📋 Credential Summary:**
- **System User:** adetayo / Adebayo@120
- **Database User:** glyco_admin / glyco_secure_pass_2025
- **MinIO Keys:** glyco_admin / glyco_secure_pass_2025
- **Jupyter Token:** glycoinfo

**🔒 Keep this document secure and update as credentials change!**