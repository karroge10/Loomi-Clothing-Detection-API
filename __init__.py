"""
Loomi Clothing Detection API
============================

A clean, modular FastAPI application for AI-powered clothing analysis and segmentation.

Features:
- Async model loading
- Rate limiting and user management
- Queue-based processing
- Token system foundation
- Clean, maintainable code structure

Modules:
- main.py: Main FastAPI application
- config.py: Configuration management
- rate_limiter.py: Rate limiting and user tracking
- model_manager.py: Async ML model management
- request_queue.py: Background task processing
- user_manager.py: User identification and management
- clothing_detector.py: Core ML inference
- process.py: Image processing utilities

Author: Loomi Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Loomi Team"
__description__ = "AI-powered clothing analysis and segmentation API"
