from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
import base64
from typing import Optional

# Import our modules
from rate_limiter import rate_limiter
from config import config
from user_manager import get_user_id

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loomi Clothing Detection API", 
    description="AI-powered clothing analysis and segmentation API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "name": "Loomi Clothing Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/clothing",
            "/analyze", 
            "/analyze/download"
        ],
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/clothing")
async def detect_clothing(
    file: UploadFile = File(...),
    request: Request = None
):
    """
    Detect clothing types in the uploaded image.
    
    Returns:
    - clothing_types: List of detected clothing types with confidence scores
    - processing_time: Time taken for detection
    """
    try:
        # Get user ID for rate limiting
        logger.info("Getting user ID for rate limiting...")
        user_id = get_user_id(request)
        logger.info(f"User ID obtained: {user_id}")
        
        # Check rate limit
        logger.info("Checking rate limit...")
        if not await rate_limiter.check_rate_limit(user_id, "/clothing"):
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
        await rate_limiter.add_request(user_id, "/clothing", len(image_bytes))
        
        # Use the proper clothing detector from clothing_detector.py
        logger.info("Importing clothing detector...")
        from clothing_detector import detect_clothing_types
        
        logger.info("Starting clothing detection...")
        clothing_result = detect_clothing_types(image_bytes)
        logger.info("Clothing detection completed successfully")
        
        # Remove request from concurrent tracking
        logger.info("Removing request from concurrent tracking...")
        await rate_limiter.remove_request(user_id)
        
        return JSONResponse(clothing_result)
        
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
    file: UploadFile = File(...),
    selected_clothing: Optional[str] = Form(None),
    request: Request = None
):
    """
    Full image analysis: clothing detection, clothing-only image, dominant color.
    
    - selected_clothing: Optional clothing type to focus on
    - color: Dominant color of clothing
    - clothing_analysis: Detected clothing types with stats
    - clothing_only_image: Base64 PNG with transparent background
    """
    try:
        # Get user ID for rate limiting
        user_id = get_user_id(request)
        
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
        
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content once
        image_bytes = await file.read()
        
        # Add request to tracking after successful validation
        await rate_limiter.add_request(user_id, "/analyze", len(image_bytes))
        
        # Use the proper clothing detector from clothing_detector.py
        from clothing_detector import detect_clothing_types, create_clothing_only_image
        from process import get_dominant_color_from_base64
        
        # Step 1: Detect clothing types
        clothing_result = detect_clothing_types(image_bytes)
        
        # Step 2: Create clothing-only image
        clothing_only_image = create_clothing_only_image(image_bytes, selected_clothing)
        
        # Step 3: Get dominant color from clothing-only image
        color = get_dominant_color_from_base64(clothing_only_image)
        
        # Remove request from concurrent tracking
        await rate_limiter.remove_request(user_id)
        
        return JSONResponse({
            "dominant_color": color,
            "clothing_analysis": clothing_result,
            "clothing_only_image": clothing_only_image,
            "selected_clothing": selected_clothing
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Remove request from concurrent tracking on error
        if 'user_id' in locals():
            await rate_limiter.remove_request(user_id)
        logger.error(f"Error in clothing analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clothing analysis: {str(e)}")

@app.post("/analyze/download")
async def analyze_image_download(
    file: UploadFile = File(...),
    selected_clothing: Optional[str] = Form(None),
    request: Request = None
):
    """
    Download clothing-only image with transparent background.
    
    - selected_clothing: Optional clothing type to focus on
    - Returns: PNG file with transparent background
    """
    try:
        # Get user ID for rate limiting
        user_id = get_user_id(request)
        
        # Check rate limit
        if not await rate_limiter.check_rate_limit(user_id, "/analyze/download"):
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
        
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content once
        image_bytes = await file.read()
        
        # Add request to tracking after successful validation
        await rate_limiter.add_request(user_id, "/analyze/download", len(image_bytes))
        
        # Use the proper clothing detector from clothing_detector.py
        from clothing_detector import create_clothing_only_image
        
        # Create clothing-only image
        clothing_only_image = create_clothing_only_image(image_bytes, selected_clothing)
        
        if not clothing_only_image:
            raise HTTPException(status_code=500, detail="Failed to create clothing-only image")
        
        # Decode base64 image
        if clothing_only_image.startswith("data:image"):
            base64_data = clothing_only_image.split(",", 1)[1]
        else:
            base64_data = clothing_only_image
            
        image_data = base64.b64decode(base64_data)
        
        # Create filename
        filename = f"clothing_analyzed_{selected_clothing or 'all'}_{file.filename}"
        if not filename.endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'
        
        # Remove request from concurrent tracking
        await rate_limiter.remove_request(user_id)
        
        return Response(
            content=image_data,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Remove request from concurrent tracking on error
        if 'user_id' in locals():
            await rate_limiter.remove_request(user_id)
        logger.error(f"Error in clothing analysis download: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clothing analysis download: {str(e)}")
