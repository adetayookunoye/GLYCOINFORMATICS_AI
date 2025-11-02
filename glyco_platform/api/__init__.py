# API package initialization
"""
FastAPI application for the Glycoinformatics AI Platform.

Provides REST API endpoints for all platform functionality including:
- Structure analysis
- Reasoning queries  
- Knowledge graph search
- Batch processing
- Health monitoring
"""

from .main import app

__all__ = ["app"]