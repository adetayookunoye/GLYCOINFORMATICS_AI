# Multi-stage Docker build for Glycoinformatics AI Platform API

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation  
FROM base as dependencies

# Create app directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY glyco_platform/ /app/glyco_platform/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

# Set permissions
RUN chmod +x /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
# Set the command to run the application
CMD ["uvicorn", "glyco_platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]