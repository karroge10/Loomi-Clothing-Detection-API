from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clothing Detection API", description="Simple image processing API")

@app.get("/")
def read_root():
    return {"Hello": "World!", "endpoints": ["/", "/upload", "/health"]}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Simple image upload endpoint - returns basic image info."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        
        # Basic image processing with PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Return basic info
        return JSONResponse({
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(image_bytes),
            "image_format": image.format,
            "image_mode": image.mode,
            "dimensions": {
                "width": image.width,
                "height": image.height
            },
            "message": "Image uploaded successfully! ML processing coming soon..."
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
