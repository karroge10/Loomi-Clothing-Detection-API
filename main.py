from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional
from pydantic import BaseModel
import traceback

# Import our modules
from rate_limiter import rate_limiter
from config import config
from user_manager import get_user_id

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class SegmentationAnalysisRequest(BaseModel):
    segmentation_data: dict
    selected_clothing: Optional[str] = None
    
    class Config:
        str_max_length = 10_000_000  # 10MB limit for segmentation data

# Pre-load models on startup for faster first request
logger.info("🚀 Pre-loading ML models for faster response...")
try:
    from clothing_detector import get_clothing_detector
    # Warm up the model
    detector = get_clothing_detector()
    logger.info("✅ ML models loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Could not pre-load models: {e}")

# CPU optimization for free tier
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads
logger.info("🔧 CPU optimized for free tier (4 threads)")

app = FastAPI(
    title="Loomi Clothing Detection API", 
    description="AI-powered clothing analysis and segmentation API",
    version="1.0.0"
)

# Increase request size limit for large segmentation data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# Custom exception handler for better error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler: {exc}")
    logger.error(f"Exception type: {type(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Return safe error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if str(exc) else "Unknown error occurred",
            "type": type(exc).__name__
        }
    )

@app.get("/")
def read_root():
    return {
        "name": "Loomi Clothing Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/detect",           # Main endpoint for clothing detection
            "/analyze",          # Analysis with data reuse
            "/health",           # Health check
            "/performance"       # Performance statistics
        ],
        "docs": "/docs",
        "workflow": {
            "step1": "POST /detect - upload image and get clothing types with segmentation",
            "step2": "POST /analyze - analyze selected clothing type (remove background, get color)"
        },
        "optimization_tips": [
            "Use /detect to get segmentation data",
            "Then use /analyze with this data for fast analysis",
            "This avoids re-running the ML model"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/performance")
def performance_stats():
    """Get performance statistics and cache info."""
    try:
        from clothing_detector import _cache_hits, _cache_misses, _segmentation_cache, _segmentation_data_cache
        
        # Calculate cache hit rate
        total_requests = _cache_hits + _cache_misses
        hit_rate = (_cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Get segmentation data cache stats
        seg_cache_stats = _segmentation_data_cache.get_stats()
        
        return {
            "cache_stats": {
                "hits": _cache_hits,
                "misses": _cache_misses,
                "hit_rate_percent": round(hit_rate, 2),
                "cached_images": len(_segmentation_cache)
            },
            "segmentation_cache": {
                "size": seg_cache_stats["size"],
                "max_size": seg_cache_stats["max_size"],
                "ttl_hours": seg_cache_stats["ttl_hours"]
            },
            "device_info": {
                "device": "cpu",
                "cuda_available": False,
                "cpu_threads": os.environ.get("OMP_NUM_THREADS", "4"),
                "optimization": "free_tier_cpu"
            },
            "performance_tips": [
                "Using CPU optimization for free tier",
                "Limited to 4 threads for stability",
                "Cache enabled for repeated images",
                "Models pre-loaded at startup",
                "Segmentation data cached for analyze endpoint"
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/detect")
async def detect_clothing(
    file: UploadFile = File(...),
    request: Request = None
):
    """
    Detect clothing types in the uploaded image with segmentation data.
    
    Returns:
    - clothing_types: List of detected clothing types with confidence scores
    - segmentation_data: Data for visualization and analysis
    - processing_time: Time taken for detection
    """
    try:
        # Get user ID for rate limiting
        logger.info("Getting user ID for rate limiting...")
        user_id = get_user_id(request)
        logger.info(f"User ID obtained: {user_id}")
        
        # Check rate limit
        logger.info("Checking rate limit...")
        if not await rate_limiter.check_rate_limit(user_id, "/detect"):
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Maximum {config.rate_limit_requests} requests per {config.rate_limit_window} seconds."
            )
        
        # Check concurrent limit
        logger.info("Checking concurrent limit...")
        if not await rate_limiter.check_concurrent_limit(user_id):
            raise HTTPException(
                status_code=429, 
                detail=f"Concurrent request limit exceeded. Maximum {config.max_concurrent_requests} concurrent requests."
            )
        
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content once
        logger.info("Reading file content...")
        image_bytes = await file.read()
        logger.info(f"File size: {len(image_bytes)} bytes")
        
        # Add request to tracking after successful validation
        logger.info("Adding request to rate limiter...")
        await rate_limiter.add_request(user_id, "/detect", len(image_bytes))
        
        # Use the optimized clothing detector from clothing_detector.py
        logger.info("Importing clothing detector...")
        from clothing_detector import detect_clothing_types_optimized
        
        logger.info("Starting clothing detection...")
        result = detect_clothing_types_optimized(image_bytes)
        logger.info("Clothing detection completed successfully")
        
        # Log detailed response size breakdown for debugging
        import json
        
        # Calculate size of each component
        clothing_result_size = len(json.dumps(result.get('clothing_instances', []))) / 1024
        masks_size = len(json.dumps(result.get('segmentation_data', {}).get('masks', {}))) / 1024
        pred_seg_size = len(json.dumps(result.get('segmentation_data', {}).get('pred_seg', []))) / 1024
        image_size_size = len(json.dumps(result.get('segmentation_data', {}).get('image_size', []))) / 1024
        image_hash_size = len(json.dumps(result.get('segmentation_data', {}).get('image_hash', ''))) / 1024
        
        # Calculate other fields
        other_fields = {k: v for k, v in result.items() if k not in ['clothing_instances', 'segmentation_data']}
        other_fields_size = len(json.dumps(other_fields)) / 1024
        
        # Total size
        response_json = json.dumps(result)
        response_size_kb = len(response_json.encode('utf-8')) / 1024
        
        logger.info("=== RESPONSE SIZE BREAKDOWN ===")
        logger.info(f"clothing_instances: {clothing_result_size:.1f} KB")
        logger.info(f"masks: {masks_size:.1f} KB")
        logger.info(f"pred_seg: {pred_seg_size:.1f} KB")
        logger.info(f"image_size: {image_size_size:.1f} KB")
        logger.info(f"image_hash: {image_hash_size:.1f} KB")
        logger.info(f"other_fields: {other_fields_size:.1f} KB")
        logger.info(f"TOTAL: {response_size_kb:.1f} KB")
        logger.info("================================")
        
        # Remove request from concurrent tracking
        logger.info("Removing request from concurrent tracking...")
        await rate_limiter.remove_request(user_id)
        
        return JSONResponse(result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Remove request from concurrent tracking on error
        if 'user_id' in locals():
            logger.info("Removing request from concurrent tracking due to error...")
            await rate_limiter.remove_request(user_id)
        logger.error(f"Error in clothing detection: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error in clothing detection: {str(e)}")

@app.post("/analyze")
async def analyze_image(
    request: SegmentationAnalysisRequest,
    http_request: Request = None
):
    """
    Analyze image using pre-computed segmentation data.
    Much faster than full analysis.
    """
    try:
        # Get user ID for rate limiting
        user_id = get_user_id(http_request)
        
        # Check rate limit
        if not await rate_limiter.check_rate_limit(user_id, "/analyze"):
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Maximum {config.rate_limit_requests} requests per {config.rate_limit_window} seconds."
            )
        
        # Check concurrent limit
        if not await rate_limiter.check_concurrent_limit(user_id):
            raise HTTPException(
                status_code=429, 
                detail=f"Concurrent request limit exceeded. Maximum {config.max_concurrent_requests} concurrent requests."
            )
        
        # Add request to tracking
        await rate_limiter.add_request(user_id, "/analyze", 0)
        
        # Use pre-computed segmentation data
        from clothing_detector import analyze_from_segmentation
        
        result = analyze_from_segmentation(request.segmentation_data, request.selected_clothing)
        
        # Remove request from concurrent tracking
        await rate_limiter.remove_request(user_id)
        
        return JSONResponse(result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Remove request from concurrent tracking on error
        if 'user_id' in locals():
            await rate_limiter.remove_request(user_id)
        logger.error(f"Error in analysis with segmentation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in analysis with segmentation: {str(e)}")
