from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from process import get_dominant_color_from_base64
from clothing_detector import (
    detect_clothing_types,
    create_clothing_only_image,
    get_clothing_detector,
)
import logging
import os
import base64
from starlette import status

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FashionAI API", description="Clothing analysis & segmentation API")

# CORS (configure with env ALLOWED_ORIGINS="http://localhost:5173,https://your-site")
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins: List[str]
if allowed_origins_env.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API settings
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    c.strip() for c in os.getenv("ALLOWED_CONTENT_TYPES", "image/jpeg,image/png,image/webp").split(",") if c.strip()
}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error"},
    )

@app.on_event("startup")
async def maybe_warmup_model():
    if os.getenv("WARMUP_ON_STARTUP", "true").lower() in {"1", "true", "yes"}:
        # Warm up model on startup to reduce first request latency
        get_clothing_detector()


@app.get("/")
async def api_root():
    return JSONResponse({
        "name": "FashionAI API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/clothing", "/analyze", "/analyze/base64", "/labels", "/healthz"],
    })


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.post("/clothing")
async def get_clothing_list(file: UploadFile = File(...)):
    """Detect all clothing types on image and return coordinates."""
    logger.info(f"Processing clothing detection for file: {file.filename}")
    # Validation
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content-type: {file.content_type}")
    # Read with size guard
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB")
    clothing_result = detect_clothing_types(image_bytes)
    logger.info(f"Clothing detection completed. Found {clothing_result.get('total_detected', 0)} items")
    return clothing_result

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
    logger.info(f"Processing full analysis for file: {file.filename}, selected_clothing: {selected_clothing}")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content-type: {file.content_type}")
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB")

    # Step 1: Detect clothing types (cached segmentation)
    logger.info("Detecting clothing types...")
    clothing_result = detect_clothing_types(image_bytes)

    # Step 2: Create clothing-only image (cached segmentation)
    logger.info("Creating clothing-only image...")
    clothing_only_image = create_clothing_only_image(image_bytes, selected_clothing)

    # Step 3: Get dominant color from clothing-only image (no background)
    logger.info("Getting dominant color from clothing-only image...")
    color = get_dominant_color_from_base64(clothing_only_image)

    logger.info("Full analysis completed successfully")
    return JSONResponse(content={
        "dominant_color": color,
        "clothing_analysis": clothing_result,
        "clothing_only_image": clothing_only_image,
        "selected_clothing": selected_clothing
    })


class Base64AnalyzeRequest(BaseModel):
    image_base64: str
    selected_clothing: Optional[str] = None


@app.post("/analyze/base64")
async def analyze_image_base64(payload: Base64AnalyzeRequest):
    """Analyze base64-encoded image (handy for React Native)."""
    # Decode image from base64
    if payload.image_base64.startswith("data:image"):
        base64_data = payload.image_base64.split(",", 1)[1]
    else:
        base64_data = payload.image_base64

    image_bytes = base64.b64decode(base64_data)

    # 1) Clothing detection
    clothing_result = detect_clothing_types(image_bytes)

    # 2) Clothing-only image
    clothing_only_image = create_clothing_only_image(image_bytes, payload.selected_clothing)

    # 3) Dominant color from clothing-only image
    color = get_dominant_color_from_base64(clothing_only_image)

    return JSONResponse(content={
        "dominant_color": color,
        "clothing_analysis": clothing_result,
        "clothing_only_image": clothing_only_image,
        "selected_clothing": payload.selected_clothing,
    })


@app.get("/labels")
async def get_labels():
    detector = get_clothing_detector()
    return {"labels": list(detector.labels.values())}