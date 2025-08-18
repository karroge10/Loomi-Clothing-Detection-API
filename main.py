from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import logging
import base64
import numpy as np
from sklearn.cluster import KMeans

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clothing Detection API", description="Image processing API with basic color analysis")

@app.get("/")
def read_root():
    return {
        "Hello": "World!", 
        "endpoints": ["/", "/upload", "/health", "/resize", "/convert", "/colors", "/colors/advanced"]
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

@app.post("/colors/advanced")
async def advanced_color_analysis(
    file: UploadFile = File(...),
    num_colors: int = Form(5)
):
    """Advanced color analysis using K-means clustering to find dominant colors."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image for faster processing (optional)
        image_small = image.resize((150, 150), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and reshape for clustering
        img_array = np.array(image_small)
        pixels = img_array.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_colors, n_init='auto', random_state=42)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # Count pixels in each cluster
        labels = kmeans.labels_
        color_counts = np.bincount(labels)
        
        # Sort colors by frequency (most common first)
        sorted_indices = np.argsort(color_counts)[::-1]
        dominant_colors = dominant_colors[sorted_indices]
        color_counts = color_counts[sorted_indices]
        
        # Calculate percentages
        total_pixels = len(pixels)
        color_percentages = (color_counts / total_pixels * 100).round(2)
        
        # Format results
        color_results = []
        for i, (color, count, percentage) in enumerate(zip(dominant_colors, color_counts, color_percentages)):
            r, g, b = color
            color_results.append({
                "rank": i + 1,
                "rgb": [int(r), int(g), int(b)],
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "percentage": float(percentage),
                "pixel_count": int(count)
            })
        
        # Get the most dominant color
        most_dominant = color_results[0]
        
        return JSONResponse({
            "message": f"Advanced color analysis completed with {num_colors} dominant colors",
            "most_dominant_color": {
                "rgb": most_dominant["rgb"],
                "hex": most_dominant["hex"],
                "percentage": most_dominant["percentage"]
            },
            "all_dominant_colors": color_results,
            "image_info": {
                "original_dimensions": {"width": image.width, "height": image.height},
                "processed_dimensions": {"width": 150, "height": 150},
                "total_pixels_analyzed": total_pixels
            },
            "note": "Colors are sorted by frequency. Next step: clothing segmentation!"
        })
        
    except Exception as e:
        logger.error(f"Error in advanced color analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in advanced color analysis: {str(e)}")
