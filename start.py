#!/usr/bin/env python3
"""
Startup script for Loomi Clothing Detection API
Supports different deployment scenarios and configurations
"""

import os
import sys
import argparse
import uvicorn
from config import config

def main():
    parser = argparse.ArgumentParser(description="Start Loomi Clothing Detection API")
    parser.add_argument("--host", default=config.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=config.workers, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--config", help="Path to .env file")
    parser.add_argument("--huggingface", action="store_true", help="Optimize for Hugging Face Spaces")
    
    args = parser.parse_args()
    
    # Load environment variables if specified
    if args.config:
        from dotenv import load_dotenv
        load_dotenv(args.config)
        # Reload config after loading .env
        from importlib import reload
        import config
        reload(config)
        config = config.config
    
    # Hugging Face Spaces optimization
    if args.huggingface:
        print("🚀 Optimizing for Hugging Face Spaces...")
        os.environ["WORKERS"] = "1"
        os.environ["NUM_WORKERS"] = "1"
        os.environ["MODEL_WARMUP_ON_STARTUP"] = "true"
        args.workers = 1
        args.reload = False
    
    # Validate configuration
    warnings = config.validate()
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()
    
    # Print startup information
    print("🎯 Loomi Clothing Detection API")
    print(f"   Version: {config.version}")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Background workers: {config.num_workers}")
    print(f"   Rate limit: {config.rate_limit_requests} req/min")
    print(f"   Concurrent limit: {config.max_concurrent_requests}")
    print(f"   File size limit: {config.max_upload_mb}MB")
    print(f"   Hugging Face Space: {config.is_huggingface_space}")
    print()
    
    # Start the server
    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=config.log_level.lower(),
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
