from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
import base64

import os
import uuid
import numpy as np

# Conditional import for rembg (only when needed)
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def get_dominant_color(processed_bytes, k=3):
    # Step 1: load transparent image
    image = Image.open(BytesIO(processed_bytes)).convert("RGBA")
    image = image.resize((100, 100))  # Resize to speed up

    # Step 2: Filter only visible (non-transparent) pixels
    np_image = np.array(image)
    rgb_pixels = np_image[...,:3]    # Ignore alpha channel
    alpha = np_image[..., 3]
    rgb_pixels = rgb_pixels[alpha > 0]  # Keep only pixels where alpha > 0

    # Step 3: KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(rgb_pixels)
    dominant_color = kmeans.cluster_centers_[0]
    r, g, b = map(int, dominant_color)
    return f"rgb({r}, {g}, {b})"


def get_dominant_color_from_base64(base64_image, k=3):
    """Compute dominant color from base64-encoded clothing-only image."""
    try:
        # Step 1: Decode base64 to bytes
        if base64_image.startswith('data:image'):
            # Remove data URL prefix
            base64_data = base64_image.split(',')[1]
        else:
            base64_data = base64_image
            
        image_bytes = base64.b64decode(base64_data)
        
        # Step 2: Load image and convert to RGBA
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        image = image.resize((100, 100))  # Resize to speed up

        # Step 3: Filter only visible (non-transparent) pixels
        np_image = np.array(image)
        rgb_pixels = np_image[...,:3]    # Ignore alpha channel
        alpha = np_image[..., 3]
        rgb_pixels = rgb_pixels[alpha > 0]  # Keep only pixels where alpha > 0

        # Check if we have any visible pixels
        if len(rgb_pixels) == 0:
            return "rgb(0, 0, 0)"  # Fallback to black if no visible pixels

        # Step 4: KMeans clustering
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(rgb_pixels)
        dominant_color = kmeans.cluster_centers_[0]
        r, g, b = map(int, dominant_color)
        return f"rgb({r}, {g}, {b})"
        
    except Exception as e:
        print(f"Error in get_dominant_color_from_base64: {e}")
        return "rgb(0, 0, 0)"  # Fallback to black on error


def remove_background(image_bytes: bytes) -> bytes:
    if not REMBG_AVAILABLE:
        raise ImportError("rembg module not available. Install with: pip install rembg")
    
    result_bytes = remove(image_bytes)

    # Save image to disk
    output_image = Image.open(BytesIO(result_bytes))
    file_name = f"{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join("results", file_name)
    output_image.save(output_path)

    print(f"✅ Saved background-removed image to: {output_path}")
    return result_bytes