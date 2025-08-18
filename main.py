from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import logging
import base64
import numpy as np
from sklearn.cluster import KMeans
import cv2
import tempfile
import os
import torch

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clothing Detection API", description="Image processing API with basic color analysis")

@app.get("/")
def read_root():
    return {
        "Hello": "World!", 
        "endpoints": [
            "/", "/upload", "/health", "/resize", "/convert", 
            "/colors", "/colors/advanced", 
            "/clothing/simple", "/clothing/analyze",
            "/clothing/simple/download", "/clothing/analyze/download",
            "/clothing/ml"
        ]
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

@app.post("/clothing/simple")
async def simple_clothing_detection(
    file: UploadFile = File(...),
    threshold: float = Form(0.1)
):
    """Simple clothing detection using color-based background removal."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        # Convert PIL to OpenCV format
        pil_image = Image.open(BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-background areas
        # This is a simple approach - we'll improve it in next steps
        lower_bound = np.array([0, 0, int(255 * threshold)])
        upper_bound = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = (cv_image.shape[0] * cv_image.shape[1]) * 0.01  # 1% of image
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Create clothing-only image
        clothing_mask = np.zeros_like(mask)
        cv2.drawContours(clothing_mask, valid_contours, -1, 255, -1)
        
        # Apply mask to original image
        clothing_only = cv2.bitwise_and(cv_image, cv_image, mask=clothing_mask)
        
        # Convert back to PIL for base64 encoding
        clothing_only_rgb = cv2.cvtColor(clothing_only, cv2.COLOR_BGR2RGB)
        clothing_pil = Image.fromarray(clothing_only_rgb)
        
        # Convert to base64
        buffer = BytesIO()
        clothing_pil.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Calculate clothing area percentage
        total_pixels = cv_image.shape[0] * cv_image.shape[1]
        clothing_pixels = np.sum(clothing_mask > 0)
        clothing_percentage = (clothing_pixels / total_pixels * 100).round(2)
        
        # Get bounding boxes for detected clothing areas
        clothing_areas = []
        for i, contour in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            clothing_areas.append({
                "id": i + 1,
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area_pixels": int(area),
                "area_percentage": round((area / total_pixels * 100), 2)
            })
        
        return JSONResponse({
            "message": "Simple clothing detection completed",
            "clothing_percentage": clothing_percentage,
            "total_clothing_areas": len(clothing_areas),
            "clothing_areas": clothing_areas,
            "clothing_only_image": f"data:image/png;base64,{img_base64}",
            "image_info": {
                "original_dimensions": {"width": cv_image.shape[1], "height": cv_image.shape[0]},
                "clothing_pixels": int(clothing_pixels),
                "total_pixels": int(total_pixels)
            },
            "note": "This is basic detection. Advanced ML segmentation coming in next steps!"
        })
        
    except Exception as e:
        logger.error(f"Error in simple clothing detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error in simple clothing detection: {str(e)}")

@app.post("/clothing/analyze")
async def analyze_clothing_with_colors(
    file: UploadFile = File(...),
    num_colors: int = Form(5)
):
    """Combine clothing detection with advanced color analysis."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        # First, detect clothing areas
        pil_image = Image.open(BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Simple background removal
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 255, 255]))
        
        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to get clothing-only image
        clothing_only = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        
        # Convert to PIL for color analysis
        clothing_rgb = cv2.cvtColor(clothing_only, cv2.COLOR_BGR2RGB)
        clothing_pil = Image.fromarray(clothing_rgb)
        
        # Resize for faster processing
        clothing_small = clothing_pil.resize((150, 150), Image.Resampling.LANCZOS)
        
        # Color analysis on clothing-only image
        img_array = np.array(clothing_small)
        pixels = img_array.reshape(-1, 3)
        
        # Filter out black/transparent pixels
        valid_pixels = pixels[np.any(pixels > 10, axis=1)]
        
        if len(valid_pixels) == 0:
            return JSONResponse({
                "message": "No clothing detected or image too dark",
                "clothing_percentage": 0,
                "dominant_colors": []
            })
        
        # K-means clustering on clothing pixels only
        kmeans = KMeans(n_clusters=min(num_colors, len(valid_pixels)), n_init='auto', random_state=42)
        kmeans.fit(valid_pixels)
        
        # Get results
        dominant_colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        color_counts = np.bincount(labels)
        
        # Sort by frequency
        sorted_indices = np.argsort(color_counts)[::-1]
        dominant_colors = dominant_colors[sorted_indices]
        color_counts = color_counts[sorted_indices]
        
        # Calculate percentages
        total_clothing_pixels = len(valid_pixels)
        color_percentages = (color_counts / total_clothing_pixels * 100).round(2)
        
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
        
        # Calculate clothing area
        total_pixels = cv_image.shape[0] * cv_image.shape[1]
        clothing_pixels = np.sum(mask > 0)
        clothing_percentage = (clothing_pixels / total_pixels * 100).round(2)
        
        return JSONResponse({
            "message": f"Clothing analysis completed with {len(color_results)} dominant colors",
            "clothing_percentage": clothing_percentage,
            "most_dominant_color": color_results[0] if color_results else None,
            "all_dominant_colors": color_results,
            "image_info": {
                "original_dimensions": {"width": cv_image.shape[1], "height": cv_image.shape[0]},
                "clothing_pixels": int(clothing_pixels),
                "total_pixels": int(total_pixels),
                "analyzed_clothing_pixels": int(total_clothing_pixels)
            },
            "note": "Colors analyzed only on detected clothing areas. ML segmentation next!"
        })
        
    except Exception as e:
        logger.error(f"Error in clothing analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clothing analysis: {str(e)}")

@app.post("/clothing/simple/download")
async def simple_clothing_detection_download(
    file: UploadFile = File(...),
    threshold: float = Form(0.1)
):
    """Simple clothing detection - returns downloadable image."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        # Convert PIL to OpenCV format
        pil_image = Image.open(BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for non-background areas
        lower_bound = np.array([0, 0, int(255 * threshold)])
        upper_bound = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = (cv_image.shape[0] * cv_image.shape[1]) * 0.01  # 1% of image
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Create clothing-only image
        clothing_mask = np.zeros_like(mask)
        cv2.drawContours(clothing_mask, valid_contours, -1, 255, -1)
        
        # Apply mask to original image
        clothing_only = cv2.bitwise_and(cv_image, cv_image, mask=clothing_mask)
        
        # Convert back to PIL for saving
        clothing_only_rgb = cv2.cvtColor(clothing_only, cv2.COLOR_BGR2RGB)
        clothing_pil = Image.fromarray(clothing_only_rgb)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            clothing_pil.save(tmp_file.name, format="PNG")
            tmp_file_path = tmp_file.name
        
        # Return file for download
        return FileResponse(
            tmp_file_path,
            media_type="image/png",
            filename=f"clothing_detected_{file.filename}",
            background=lambda: os.unlink(tmp_file_path)  # Clean up after sending
        )
        
    except Exception as e:
        logger.error(f"Error in simple clothing detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error in simple clothing detection: {str(e)}")

@app.post("/clothing/analyze/download")
async def analyze_clothing_with_colors_download(
    file: UploadFile = File(...),
    num_colors: int = Form(5)
):
    """Combine clothing detection with color analysis - returns downloadable image."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        # First, detect clothing areas
        pil_image = Image.open(BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Simple background removal
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 255, 255]))
        
        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to get clothing-only image
        clothing_only = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        
        # Convert to PIL for saving
        clothing_rgb = cv2.cvtColor(clothing_only, cv2.COLOR_BGR2RGB)
        clothing_pil = Image.fromarray(clothing_rgb)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            clothing_pil.save(tmp_file.name, format="PNG")
            tmp_file_path = tmp_file.name
        
        # Return file for download
        return FileResponse(
            tmp_file_path,
            media_type="image/png",
            filename=f"clothing_analyzed_{file.filename}",
            background=lambda: os.unlink(tmp_file_path)  # Clean up after sending
        )
        
    except Exception as e:
        logger.error(f"Error in clothing analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in clothing analysis: {str(e)}")

@app.post("/clothing/ml")
async def ml_clothing_detection(
    file: UploadFile = File(...)
):
    """ML-based clothing detection using transformers and proper segmentation."""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        # Load image with PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Load the clothing segmentation model
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        
        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        
        # Prepare inputs
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # Get predicted segmentation
        pred_seg = logits.argmax(dim=1)[0].numpy()
        
        # Clothing labels mapping
        clothing_labels = {
            0: "Background",
            1: "Hat", 
            2: "Hair",
            3: "Sunglasses",
            4: "Upper-clothes",
            5: "Skirt",
            6: "Pants",
            7: "Dress",
            8: "Belt",
            9: "Left-shoe",
            10: "Right-shoe",
            11: "Face",
            12: "Left-leg",
            13: "Right-leg",
            14: "Left-arm",
            15: "Right-arm",
            16: "Bag",
            17: "Scarf"
        }
        
        # Clothing classes (exclude body parts and background)
        clothing_classes = [4, 5, 6, 7, 8, 9, 10, 16, 17]
        
        # Create clothing mask
        clothing_mask = np.isin(pred_seg, clothing_classes)
        
        # Convert to OpenCV format for processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply mask to get clothing-only image
        clothing_only = cv_image.copy()
        clothing_only[~clothing_mask] = [0, 0, 0]  # Set non-clothing to black
        
        # Convert back to PIL for saving
        clothing_only_rgb = cv2.cvtColor(clothing_only, cv2.COLOR_BGR2RGB)
        clothing_pil = Image.fromarray(clothing_only_rgb)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            clothing_pil.save(tmp_file.name, format="PNG")
            tmp_file_path = tmp_file.name
        
        # Calculate statistics
        total_pixels = clothing_mask.shape[0] * clothing_mask.shape[1]
        clothing_pixels = np.sum(clothing_mask)
        clothing_percentage = (clothing_pixels / total_pixels * 100).round(2)
        
        # Count detected clothing types
        detected_clothing = {}
        for class_id in clothing_classes:
            if class_id in pred_seg:
                count = np.sum(pred_seg == class_id)
                if count > 0:
                    detected_clothing[clothing_labels[class_id]] = {
                        "pixels": int(count),
                        "percentage": round((count / total_pixels * 100), 2)
                    }
        
        # Return file for download
        return FileResponse(
            tmp_file_path,
            media_type="image/png",
            filename=f"ml_clothing_detected_{file.filename}",
            background=lambda: os.unlink(tmp_file_path)  # Clean up after sending
        )
        
    except Exception as e:
        logger.error(f"Error in ML clothing detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error in ML clothing detection: {str(e)}")
