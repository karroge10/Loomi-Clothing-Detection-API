#!/usr/bin/env python3
"""
Simple startup script for Loomi Clothing Detection API
"""

import uvicorn
from config import config

if __name__ == "__main__":
    print("🎯 Loomi Clothing Detection API")
    print(f"   Version: {config.version}")
    print(f"   Host: {config.host}:{config.port}")
    print(f"   Workers: {config.workers}")
    print(f"   File size limit: {config.max_upload_mb}MB")
    print(f"   Hugging Face Space: {config.is_huggingface_space}")
    print()
    
    try:
        uvicorn.run(
            "main:app",
            host=config.host,
            port=config.port,
            workers=config.workers,
            log_level=config.log_level.lower(),
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        exit(1)
