from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
import base64
from typing import Optional

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
async def get_clothing_list(file: UploadFile = File(...)):
    """Detect all clothing types on image and return coordinates."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        # Use the proper clothing detector from clothing_detector.py
        from clothing_detector import detect_clothing_types
        clothing_result = detect_clothing_types(image_bytes)
        
        return JSONResponse(clothing_result)
        
    except Exception as e:
        logger.error(f"Error in clothing detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clothing detection: {str(e)}")

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    selected_clothing: Optional[str] = Form(None)
):
    """
    Full image analysis: clothing detection, clothing-only image, dominant color.
    
    - selected_clothing: Optional clothing type to focus on
    - color: Dominant color of clothing
    - clothing_analysis: Detected clothing types with stats
    - clothing_only_image: Base64 PNG with transparent background
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        # Use the proper clothing detector from clothing_detector.py
        from clothing_detector import detect_clothing_types, create_clothing_only_image
        from process import get_dominant_color_from_base64
        
        # Step 1: Detect clothing types
        clothing_result = detect_clothing_types(image_bytes)
        
        # Step 2: Create clothing-only image
        clothing_only_image = create_clothing_only_image(image_bytes, selected_clothing)
        
        # Step 3: Get dominant color from clothing-only image
        color = get_dominant_color_from_base64(clothing_only_image)
        
        return JSONResponse({
            "dominant_color": color,
            "clothing_analysis": clothing_result,
            "clothing_only_image": clothing_only_image,
            "selected_clothing": selected_clothing
        })
        
    except Exception as e:
        logger.error(f"Error in clothing analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clothing analysis: {str(e)}")

@app.post("/analyze/download")
async def analyze_image_download(
    file: UploadFile = File(...),
    selected_clothing: Optional[str] = Form(None)
):
    """
    Download clothing-only image with transparent background.
    
    - selected_clothing: Optional clothing type to focus on
    - Returns: PNG file with transparent background
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
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
        
        return Response(
            content=image_data,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error in clothing analysis download: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clothing analysis download: {str(e)}")
