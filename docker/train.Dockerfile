# Dockerfile for GPU-accelerated training
# Base image with PyTorch, Python 3.10, and CUDA 11.8
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 1. System Setup
# ===============
WORKDIR /app

# Set non-interactive frontend to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install git and other essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python Environment
# =====================
# Copy requirements first to leverage Docker layer caching
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the specific versions that we found were necessary during debugging
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    huggingface_hub==0.20.3 \
    pyarrow==14.0.2 \
    pandas==2.1.4 \
    numpy==1.26.2

# 3. Application Code
# ===================
# Copy the rest of the application code
COPY . /app

# Install the project in editable mode
RUN pip install -e .

# 4. Entrypoint
# =============
# The command to run when the container starts.
# This will execute the training script.
ENTRYPOINT ["python", "scripts/integrate_glycoworks_training.py"]
