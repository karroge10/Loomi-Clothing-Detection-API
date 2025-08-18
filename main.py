from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import logging
import base64
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clothing Detection API", description="Image processing API with basic color analysis")

@app.get("/")
def read_root():
    return {
        "Hello": "World!", 
        "endpoints": ["/", "/upload", "/health", "/resize", "/convert", "/colors"]
    }

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

@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Form(100),
    height: int = Form(100)
):
    """Resize image to specified dimensions."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Resize image
        resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert to base64 for response
        buffer = BytesIO()
        resized_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "message": f"Image resized to {width}x{height}",
            "original_dimensions": {"width": image.width, "height": image.height},
            "new_dimensions": {"width": width, "height": height},
            "resized_image": f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error resizing image: {str(e)}")

@app.post("/convert")
async def convert_image(
    file: UploadFile = File(...),
    format: str = Form("PNG")
):
    """Convert image to specified format."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convert format
        buffer = BytesIO()
        image.save(buffer, format=format.upper())
        converted_bytes = buffer.getvalue()
        
        # Convert to base64
        img_base64 = base64.b64encode(converted_bytes).decode()
        
        return JSONResponse({
            "message": f"Image converted to {format.upper()}",
            "original_format": image.format,
            "new_format": format.upper(),
            "new_size_bytes": len(converted_bytes),
            "converted_image": f"data:image/{format.lower()};base64,{img_base64}"
        })
        
    except Exception as e:
        logger.error(f"Error converting image: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting image: {str(e)}")

@app.post("/colors")
async def analyze_colors(file: UploadFile = File(...)):
    """Basic color analysis - average RGB values."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate average colors
        avg_r = int(np.mean(img_array[:, :, 0]))
        avg_g = int(np.mean(img_array[:, :, 1]))
        avg_b = int(np.mean(img_array[:, :, 2]))
        
        # Calculate dominant color (simplified - just brightest channel)
        max_channel = np.argmax([avg_r, avg_g, avg_b])
        dominant_colors = ["Red", "Green", "Blue"]
        
        return JSONResponse({
            "message": "Basic color analysis completed",
            "average_colors": {
                "red": avg_r,
                "green": avg_g,
                "blue": avg_b,
                "rgb": f"rgb({avg_r}, {avg_g}, {avg_b})"
            },
            "dominant_channel": dominant_colors[max_channel],
            "brightness": int((avg_r + avg_g + avg_b) / 3),
            "note": "This is basic analysis. Advanced ML color detection coming in next steps!"
        })
        
    except Exception as e:
        logger.error(f"Error analyzing colors: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing colors: {str(e)}")
