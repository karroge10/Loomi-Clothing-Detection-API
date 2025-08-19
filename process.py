from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
import base64
import hashlib
import time

import os
import uuid
import numpy as np

# Import rembg for background removal
from rembg import remove
REMBG_AVAILABLE = True

# Cache for dominant colors (image_hash -> color_result)
_color_cache = {}
_cache_ttl = 3600  # 1 hour TTL

def _get_image_hash_from_base64(base64_image):
    """Create hash from base64 image for caching."""
    if base64_image.startswith('data:image'):
        base64_data = base64_image.split(',')[1]
    else:
        base64_data = base64_image
    return hashlib.md5(base64_data.encode()).hexdigest()

def _cleanup_color_cache():
    """Remove expired cache entries."""
    global _color_cache
    current_time = time.time()
    expired_keys = [
        key for key, (_, timestamp) in _color_cache.items()
        if current_time - timestamp > _cache_ttl
    ]
    for key in expired_keys:
        del _color_cache[key]
    if expired_keys:
        print(f"Cleaned up {len(expired_keys)} expired color cache entries")

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
    """Compute dominant color from base64-encoded clothing-only image with caching."""
    try:
        # Check cache first
        image_hash = _get_image_hash_from_base64(base64_image)
        if image_hash in _color_cache:
            color_result, timestamp = _color_cache[image_hash]
            if time.time() - timestamp < _cache_ttl:
                print(f"🎨 Using cached color result for hash: {image_hash[:8]}...")
                return color_result
        
        print(f"🎨 Computing dominant color for new image (hash: {image_hash[:8]}...)")
        
        # Step 1: Decode base64 to bytes
        if base64_image.startswith('data:image'):
            # Remove data URL prefix
            base64_data = base64_image.split(',')[1]
        else:
            base64_data = base64_image
            
        image_bytes = base64.b64decode(base64_data)
        
        # Step 2: Load image and convert to RGBA
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        
        # Step 3: Optimize size for faster processing
        # Use smaller size for very large images, but keep reasonable quality
        if image.width > 200 or image.height > 200:
            # Calculate optimal size (balance between speed and quality)
            max_dim = max(image.width, image.height)
            if max_dim > 1000:
                target_size = (150, 150)  # Very large images
            elif max_dim > 500:
                target_size = (200, 200)  # Large images
            else:
                target_size = (100, 100)  # Medium images
            
            image = image.resize(target_size, Image.LANCZOS)
            print(f"🔄 Resized image from {image.width}x{image.height} to {target_size[0]}x{target_size[1]} for faster processing")
        else:
            # Small images - resize to standard size for consistency
            image = image.resize((100, 100), Image.LANCZOS)

        # Step 4: Filter only visible (non-transparent) pixels
        np_image = np.array(image)
        rgb_pixels = np_image[...,:3]    # Ignore alpha channel
        alpha = np_image[..., 3]
        rgb_pixels = rgb_pixels[alpha > 0]  # Keep only pixels where alpha > 0

        # Check if we have any visible pixels
        if len(rgb_pixels) == 0:
            result = "rgb(0, 0, 0)"  # Fallback to black if no visible pixels
        else:
            # Step 5: Optimized KMeans clustering
            # Use fewer clusters for faster processing on smaller datasets
            actual_k = min(k, len(rgb_pixels) // 10)  # Ensure we have enough pixels per cluster
            if actual_k < 1:
                actual_k = 1
            
            # Use faster KMeans settings
            kmeans = KMeans(
                n_clusters=actual_k, 
                n_init=1,  # Single initialization for speed
                max_iter=100,  # Limit iterations
                random_state=42  # Deterministic results
            )
            kmeans.fit(rgb_pixels)
            dominant_color = kmeans.cluster_centers_[0]
            r, g, b = map(int, dominant_color)
            result = f"rgb({r}, {g}, {b})"
        
        # Cache the result
        _color_cache[image_hash] = (result, time.time())
        _cleanup_color_cache()  # Clean up expired entries
        
        print(f"✅ Color analysis completed: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error in get_dominant_color_from_base64: {e}")
        return "rgb(0, 0, 0)"  # Fallback to black on error


def remove_background(image_bytes: bytes) -> bytes:
    result_bytes = remove(image_bytes)

    # Save image to disk
    output_image = Image.open(BytesIO(result_bytes))
    file_name = f"{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join("results", file_name)
    output_image.save(output_path)

    print(f"✅ Saved background-removed image to: {output_path}")
    return result_bytes