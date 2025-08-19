import hashlib
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
from io import BytesIO
import numpy as np
from collections import Counter
import logging
import base64
import warnings
import traceback

# Suppress transformers warnings for cleaner logs
warnings.filterwarnings("ignore", message=".*feature_extractor_type.*")
warnings.filterwarnings("ignore", message=".*reduce_labels.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for segmentation results with smart priorities
_segmentation_cache = {}
_cache_hits = 0
_cache_misses = 0
_cache_access_times = {}  # Track when items were last accessed

# Cache configuration
MAX_CACHE_SIZE = 15  # Increased from 10
CACHE_PRIORITY_THRESHOLD = 5  # Minimum access count to keep in cache

def _cleanup_cache():
    """Smart cache cleanup based on access patterns and memory usage."""
    global _segmentation_cache, _cache_access_times
    
    if len(_segmentation_cache) <= MAX_CACHE_SIZE:
        return
    
    # Calculate priority scores (access count * recency)
    import time
    current_time = time.time()
    
    priority_scores = {}
    for key, access_info in _cache_access_times.items():
        if key in _segmentation_cache:
            recency = current_time - access_info['last_access']
            priority = access_info['access_count'] / (1 + recency / 3600)  # Normalize by hour
            priority_scores[key] = priority
    
    # Remove lowest priority items
    items_to_remove = len(_segmentation_cache) - MAX_CACHE_SIZE
    sorted_keys = sorted(priority_scores.keys(), key=lambda k: priority_scores[k])
    
    for key in sorted_keys[:items_to_remove]:
        del _segmentation_cache[key]
        del _cache_access_times[key]
        logger.info(f"Removed low-priority cache item: {key}")
    
    logger.info(f"Cache cleaned up: {len(_segmentation_cache)} items remaining")

def _update_cache_access(image_hash: str):
    """Update cache access statistics."""
    global _cache_access_times
    import time
    
    current_time = time.time()
    if image_hash in _cache_access_times:
        _cache_access_times[image_hash]['access_count'] += 1
        _cache_access_times[image_hash]['last_access'] = current_time
    else:
        _cache_access_times[image_hash] = {
            'access_count': 1,
            'last_access': current_time
        }

# Temporarily clear cache for testing improved quality
_segmentation_cache.clear()
_cache_hits = 0
_cache_misses = 0
logger.info("Cache cleared for testing improved quality")

class ClothingDetector:
    def __init__(self):
        """Initialize clothing segmentation model."""
        self.device = torch.device("cpu")  # Force CPU for free tier
        logger.info(f"Using device: {self.device} (free tier optimization)")
        
        # Load processor and model with CPU optimizations
        logger.info("Loading SegformerImageProcessor...")
        self.processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes",
            # Remove deprecated arguments that cause warnings
        )
        
        logger.info("Loading AutoModelForSemanticSegmentation...")
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes",
            torch_dtype=torch.float32,  # Use FP32 for CPU stability
            low_cpu_mem_usage=True,  # Reduce memory usage
        )
        
        logger.info(f"Moving model to {self.device}...")
        self.model.to(self.device)
        self.model.eval()
        
        # CPU-specific optimizations
        torch.set_num_threads(4)  # Limit CPU threads for stability
        logger.info("Clothing detector initialized successfully (CPU optimized)")
        
        # Clothing labels mapping
        self.labels = {
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
        self.clothing_classes = [4, 5, 6, 7, 8, 9, 10, 16, 17]  # Upper-clothes, Skirt, Pants, Dress, Belt, Left-shoe, Right-shoe, Bag, Scarf
    
    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Create image hash to use as cache key."""
        return hashlib.md5(image_bytes).hexdigest()
    
    def _segment_image(self, image_bytes: bytes):
        """Run image segmentation with caching."""
        image_hash = self._get_image_hash(image_bytes)
        
        # Check cache (re-enabled now that quality is improved)
        if image_hash in _segmentation_cache:
            global _cache_hits
            _cache_hits += 1
            _update_cache_access(image_hash)  # Update access statistics
            logger.info("Using cached high-quality segmentation result")
            return _segmentation_cache[image_hash]
        
        global _cache_misses
        _cache_misses += 1
        # Run segmentation
        logger.info("Performing new high-quality segmentation")
        
        try:
            # Load and preprocess image
            image = Image.open(BytesIO(image_bytes))
            image = image.convert('RGB')
            
            # Prepare inputs for the model
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            pred_seg = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Upsample logits to original image size for better quality
            import torch.nn.functional as nn
            
            # Ensure logits have correct shape (N, C, H, W)
            if logits.dim() == 3:
                logits = logits.unsqueeze(0)  # Add batch dimension if missing
            
            # Get image dimensions
            img_height, img_width = image.size[1], image.size[0]  # PIL uses (width, height)
            
            # Log tensor shapes for debugging
            logger.info(f"Logits shape: {logits.shape}, Target size: ({img_height}, {img_width})")
            
            # Ensure target size is valid
            if img_height <= 0 or img_width <= 0:
                logger.warning(f"Invalid image dimensions: {img_height}x{img_width}, using original segmentation")
                pred_seg_high_quality = pred_seg
            else:
                # Upsample logits to original image size
                logits_upsampled = nn.interpolate(
                    logits,
                    size=(img_height, img_width),  # Use (height, width) format
                    mode="bilinear",
                    align_corners=False,
                )
                
                # Get high-quality predictions
                pred_seg_high_quality = logits_upsampled.argmax(dim=1)[0].cpu().numpy()
                
                logger.info(f"Created high-quality segmentation: {pred_seg_high_quality.shape} for image size {image.size}")
            
            # Store result in cache
            _segmentation_cache[image_hash] = {
                'pred_seg': pred_seg_high_quality,  # Use high-quality version
                'image': image
            }
            
            # Update cache access and cleanup if needed
            _update_cache_access(image_hash)
            _cleanup_cache()
            
            return {
                'pred_seg': pred_seg_high_quality,  # Return high-quality version
                'image': image
            }
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            raise
    
    def detect_clothing(self, image_bytes: bytes) -> dict:
        """
        Detect clothing types on image and return coordinates.

        Args:
            image_bytes: Raw image bytes

        Returns:
            dict: Clothing types with pixel stats and bounding boxes
        """
        try:
            # Get cached segmentation result
            seg_result = self._segment_image(image_bytes)
            pred_seg = seg_result['pred_seg']
            image = seg_result['image']
            
            # Count pixels per class and compute bounding boxes
            clothing_types = {}
            coordinates = {}
            total_pixels = pred_seg.size
            
            for class_id, label_name in self.labels.items():
                if label_name not in ["Background", "Face", "Hair", "Left-arm", "Right-arm", "Left-leg", "Right-leg"]:
                    # Create mask for this class
                    mask = (pred_seg == class_id)
                    
                    if np.any(mask):
                        # Count pixels
                        count = np.sum(mask)
                        percentage = (count / total_pixels) * 100
                        
                        clothing_types[label_name] = {
                            "pixels": int(count),
                            "percentage": round(percentage, 2)
                        }
                        
                        # Compute bounding box
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        
                        if np.any(rows) and np.any(cols):
                            y_min, y_max = np.where(rows)[0][[0, -1]]
                            x_min, x_max = np.where(cols)[0][[0, -1]]
                            
                            # Add padding (10% of clothing size)
                            clothing_width = x_max - x_min
                            clothing_height = y_max - y_min
                            padding_x = int(clothing_width * 0.1)
                            padding_y = int(clothing_height * 0.1)
                            
                            # Apply padding with image bounds
                            x_min = max(0, x_min - padding_x)
                            y_min = max(0, y_min - padding_y)
                            x_max = min(image.width, x_max + padding_x)
                            y_max = min(image.height, y_max + padding_y)
                            
                            coordinates[label_name] = {
                                "x_min": int(x_min),
                                "y_min": int(y_min),
                                "x_max": int(x_max),
                                "y_max": int(y_max),
                                "width": int(x_max - x_min),
                                "height": int(y_max - y_min)
                            }
            
            # Sort by percentage area
            sorted_clothing = dict(sorted(
                clothing_types.items(), 
                key=lambda x: x[1]["percentage"], 
                reverse=True
            ))
            
            # Convert to the format expected by the API
            clothing_instances = []
            for label_name, stats in sorted_clothing.items():
                if label_name in coordinates:
                    coord = coordinates[label_name]
                    clothing_instances.append({
                        "type": label_name,
                        "class_id": next(class_id for class_id, name in self.labels.items() if name == label_name),
                        "bbox": {
                            "x": coord["x_min"],
                            "y": coord["y_min"], 
                            "width": coord["width"],
                            "height": coord["height"]
                        },
                        "area_pixels": stats["pixels"],
                        "area_percentage": stats["percentage"]
                    })
            
            return {
                "message": f"Clothing detection completed. Found {len(sorted_clothing)} items",
                "total_detected": len(sorted_clothing),
                "clothing_instances": clothing_instances,
                "image_info": {
                    "width": image.width,
                    "height": image.height,
                    "total_pixels": image.width * image.height
                }
            }
            
        except Exception as e:
            logger.error(f"Error in clothing detection: {str(e)}")
            return {
                "message": "Error in clothing detection",
                "total_detected": 0,
                "clothing_instances": [],
                "image_info": {
                    "width": 0,
                    "height": 0,
                    "total_pixels": 0
                },
                "error": str(e)
            }
    
    def create_clothing_only_image(self, image_bytes: bytes, selected_clothing: str = None) -> str:
        """
        Create clothing-only image with transparent background.

        Args:
            image_bytes: Raw image bytes
            selected_clothing: Optional clothing label to isolate

        Returns:
            str: Base64-encoded PNG data URL
        """
        try:
            # Get cached segmentation
            seg_result = self._segment_image(image_bytes)
            pred_seg = seg_result['pred_seg']
            image = seg_result['image']
            
            # Create clothing-only mask
            clothing_mask = np.zeros_like(pred_seg, dtype=bool)
            
            if selected_clothing:
                # If specific clothing selected, find its class id
                selected_class_id = None
                for class_id, label_name in self.labels.items():
                    if label_name == selected_clothing:
                        selected_class_id = class_id
                        break
                
                if selected_class_id is not None:
                    # Build mask only for the selected class
                    clothing_mask = (pred_seg == selected_class_id)
                else:
                    # If not found, fall back to all clothing classes
                    for class_id in self.clothing_classes:
                        clothing_mask |= (pred_seg == class_id)
            else:
                # Otherwise, use all clothing classes
                for class_id in self.clothing_classes:
                    clothing_mask |= (pred_seg == class_id)
            
            # Convert image to numpy array
            image_array = np.array(image)
            
            # Compose RGBA with transparent background
            clothing_only_rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
            clothing_only_rgba[..., :3] = image_array  # RGB channels
            clothing_only_rgba[..., 3] = 255  # Alpha channel (opaque)
            clothing_only_rgba[~clothing_mask, 3] = 0  # Transparent for non-clothing
            
            # Create PIL image
            clothing_image = Image.fromarray(clothing_only_rgba, 'RGBA')
            
            # If a specific clothing selected, crop with padding
            if selected_clothing and selected_class_id is not None:
                clothing_image = self._crop_with_padding(clothing_image, clothing_mask)
            
            # Encode to base64
            buffer = BytesIO()
            clothing_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error in creating clothing-only image: {str(e)}")
            return ""
    
    def _crop_with_padding(self, image: Image.Image, mask: np.ndarray, padding_percent: float = 0.1) -> Image.Image:
        """
        Crop image around clothing mask with padding.

        Args:
            image: PIL image
            mask: Clothing mask
            padding_percent: Padding percentage relative to clothing size

        Returns:
            Image.Image: Cropped image
        """
        try:
            # Find clothing bounds
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return image  # If no clothing found, return original
            
            # Get bounds
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Compute clothing size
            clothing_width = x_max - x_min
            clothing_height = y_max - y_min
            
            # Compute padding
            padding_x = int(clothing_width * padding_percent)
            padding_y = int(clothing_height * padding_percent)
            
            # Apply padding within image bounds
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(image.width, x_max + padding_x)
            y_max = min(image.height, y_max + padding_y)
            
            # Crop
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            return cropped_image
            
        except Exception as e:
            logger.error(f"Error in cropping with padding: {str(e)}")
            return image

    def detect_clothing_with_segmentation(self, image_bytes: bytes) -> dict:
        """
        Detect clothing types with full segmentation data for reuse.
        Returns both clothing info and segmentation data.
        """
        try:
            seg_result = self._segment_image(image_bytes)
            pred_seg = seg_result['pred_seg']
            image = seg_result['image']
            
            clothing_result = self.detect_clothing(image_bytes)
            
            # Convert original image to base64 for reuse
            import base64
            from io import BytesIO
            
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            original_image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create highlighted images for each clothing type
            highlighted_images = {}
            
            # Add "All clothing types" highlight
            try:
                all_clothing_highlight = self.create_clothing_highlight_image(image_bytes, None)  # None = all clothing
                highlighted_images['all'] = all_clothing_highlight
                logger.info("Created highlight for all clothing types")
            except Exception as e:
                logger.warning(f"Could not create highlight for all clothing types: {e}")
                highlighted_images['all'] = original_image_base64
            
            # Create highlights for individual clothing types in parallel batches
            clothing_types = clothing_result.get('clothing_instances', [])
            if clothing_types:
                # Process in smaller batches for better memory management
                batch_size = 3
                for i in range(0, len(clothing_types), batch_size):
                    batch = clothing_types[i:i + batch_size]
                    
                    for clothing_type in batch:
                        type_name = clothing_type.get('type', '')
                        if type_name:
                            try:
                                highlighted_img = self.create_clothing_highlight_image(image_bytes, type_name)
                                highlighted_images[type_name] = highlighted_img
                                logger.info(f"Created highlight for {type_name}")
                            except Exception as e:
                                logger.warning(f"Could not create highlight for {type_name}: {e}")
                                highlighted_images[type_name] = original_image_base64
            
            # Ensure all data is JSON serializable
            return {
                **clothing_result,
                "segmentation_data": {
                    "pred_seg": pred_seg.tolist(),  # Convert numpy array to list for JSON
                    "image_size": list(image.size),  # Convert tuple to list for JSON
                    "image_hash": self._get_image_hash(image_bytes),
                    "original_image": f"data:image/png;base64,{original_image_base64}"  # Add original image
                },
                "highlighted_images": highlighted_images,  # Images with colored outlines
                "original_image": f"data:image/png;base64,{original_image_base64}"  # Original image for display
            }
        except Exception as e:
            logger.error(f"Error in clothing detection with segmentation: {e}")
            raise
    
    def analyze_from_segmentation(self, segmentation_data: dict, selected_clothing: str = None) -> dict:
        """
        Analyze image using pre-computed segmentation data.
        Much faster than full analysis.
        """
        try:
            # Reconstruct segmentation result from data
            pred_seg = np.array(segmentation_data["pred_seg"])
            image_size = segmentation_data["image_size"]
            
            # Check if we have the original image
            if "original_image" in segmentation_data:
                # Create real clothing-only image using original image
                clothing_only_image = self._create_real_clothing_only_image(
                    segmentation_data["original_image"], pred_seg, selected_clothing
                )
            else:
                # Fallback to visualization if no original image
                clothing_only_image = self._create_segmentation_visualization(
                    pred_seg, image_size, selected_clothing
                )
            
            # Get dominant color from the image
            from process import get_dominant_color_from_base64
            color = get_dominant_color_from_base64(clothing_only_image)
            
            return {
                "dominant_color": color,
                "clothing_only_image": clothing_only_image,
                "selected_clothing": selected_clothing,
                "processing_note": "Used pre-computed segmentation data with original image"
            }
            
        except Exception as e:
            logger.error(f"Error in analysis from segmentation: {e}")
            raise
    
    def _create_segmentation_visualization(self, pred_seg: np.ndarray, image_size: tuple, selected_clothing: str = None) -> str:
        """Create a visualization of the segmentation mask."""
        try:
            from PIL import Image, ImageDraw
            import base64
            from io import BytesIO
            
            # Create a new image with the segmentation visualization
            img = Image.new('RGBA', image_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Create mask for selected clothing or all clothing
            if selected_clothing:
                # Find class ID for selected clothing
                class_id = None
                for cid, label in self.labels.items():
                    if label.lower() == selected_clothing.lower():
                        class_id = cid
                        break
                
                if class_id is not None:
                    mask = (pred_seg == class_id)
                else:
                    # Fallback to all clothing if selected type not found
                    mask = np.isin(pred_seg, self.clothing_classes)
            else:
                # All clothing types
                mask = np.isin(pred_seg, self.clothing_classes)
            
            # Convert mask to PIL image
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
            
            # Create a colored overlay
            overlay = Image.new('RGBA', image_size, (100, 150, 255, 128))  # Blue with transparency
            
            # Apply mask to overlay
            overlay.putalpha(mask_img)
            
            # Composite with transparent background
            result = Image.alpha_composite(img, overlay)
            
            # Convert to base64
            buffer = BytesIO()
            result.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error creating segmentation visualization: {e}")
            # Return a simple colored square as fallback
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

    def _create_real_clothing_only_image(self, original_image_base64: str, pred_seg: np.ndarray, selected_clothing: str = None) -> str:
        """Create real clothing-only image using original image and segmentation mask."""
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            
            # Decode original image from base64
            if original_image_base64.startswith('data:image/'):
                # Remove data URL prefix
                base64_data = original_image_base64.split(',')[1]
            else:
                base64_data = original_image_base64
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            
            # Load original image
            original_image = Image.open(BytesIO(image_bytes))
            
            # Create mask for selected clothing or all clothing
            if selected_clothing:
                # Find class ID for selected clothing
                class_id = None
                for cid, label in self.labels.items():
                    if label.lower() == selected_clothing.lower():
                        class_id = cid
                        break
                
                if class_id is not None:
                    mask = (pred_seg == class_id)
                    logger.info(f"Selected clothing type '{selected_clothing}' mapped to class ID {class_id}")
                else:
                    # Fallback to all clothing if selected type not found
                    mask = np.isin(pred_seg, self.clothing_classes)
                    logger.warning(f"Could not find class ID for '{selected_clothing}', using all clothing types")
            else:
                # All clothing types
                mask = np.isin(pred_seg, self.clothing_classes)
            
            # Ensure mask and image have compatible dimensions
            mask_height, mask_width = pred_seg.shape
            img_width, img_height = original_image.size
            
            # Resize mask to match original image if needed
            if mask_height != img_height or mask_width != img_width:
                logger.info(f"Resizing mask from {mask_height}x{mask_width} to {img_height}x{img_width}")
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
                # Use LANCZOS for better quality instead of NEAREST
                mask_img = mask_img.resize((img_width, img_height), Image.LANCZOS)
                mask = np.array(mask_img) > 128  # Threshold for clean binary mask
            
            # Apply Gaussian blur to smooth the mask edges and reduce blockiness
            try:
                from scipy import ndimage
                
                # Convert to float for better precision
                mask_float = mask.astype(float)
                
                # Apply multiple smoothing passes for better quality
                # First pass: remove noise
                mask_smooth1 = ndimage.gaussian_filter(mask_float, sigma=0.8)
                
                # Second pass: smooth edges
                mask_smooth2 = ndimage.gaussian_filter(mask_smooth1, sigma=1.0)
                
                # Apply threshold with hysteresis for cleaner edges
                mask = mask_smooth2 > 0.5
                
                logger.info("Applied advanced smoothing to mask for smoother edges")
                
            except ImportError:
                logger.info("scipy not available, using basic smoothing")
                # Basic smoothing without scipy
                from PIL import ImageFilter
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.0))
                mask = np.array(mask_img) > 128
            
            # Convert original image to RGBA if it's not already
            if original_image.mode != 'RGBA':
                original_image = original_image.convert('RGBA')
            
            # Create new image with transparent background
            result = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
            
            # Apply mask to original image with smooth edges
            original_array = np.array(original_image)
            mask_array = mask.astype(np.uint8)
            
            # Create result array
            result_array = original_array.copy()
            
            # Make background transparent (where mask is 0)
            result_array[mask_array == 0, 3] = 0  # Set alpha to 0
            
            # Convert back to PIL image
            result = Image.fromarray(result_array, 'RGBA')
            
            # Convert to base64
            buffer = BytesIO()
            result.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error creating real clothing-only image: {e}")
            # Fallback to visualization
            return self._create_segmentation_visualization(pred_seg, (pred_seg.shape[1], pred_seg.shape[0]), selected_clothing)

    def create_clothing_highlight_image(self, image_bytes: bytes, selected_clothing: str = None) -> str:
        """
        Create image with highlighted selected clothing type.
        Returns base64 PNG with colored outline around selected clothing.
        """
        try:
            logger.info(f"Creating highlight image for clothing type: {selected_clothing}")
            
            seg_result = self._segment_image(image_bytes)
            pred_seg = seg_result['pred_seg']
            image = seg_result['image']
            
            logger.info(f"Segmentation shape: {pred_seg.shape}, Image size: {image.size}")
            
            # Find class ID for selected clothing
            class_id = None
            if selected_clothing:
                for cid, label in self.labels.items():
                    if label.lower() == selected_clothing.lower():
                        class_id = cid
                        break
                
                logger.info(f"Selected clothing '{selected_clothing}' mapped to class ID {class_id}")
            
            if class_id is None:
                # Use all clothing types if none selected
                mask = np.isin(pred_seg, self.clothing_classes)
                logger.info(f"Using all clothing types, mask sum: {np.sum(mask)}")
            else:
                # Use selected clothing type
                mask = (pred_seg == class_id)
                logger.info(f"Using selected clothing type {class_id}, mask sum: {np.sum(mask)}")
            
            # Create highlighted image
            from PIL import Image, ImageDraw
            
            # Convert to RGBA if needed
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Create a copy for highlighting
            highlighted_image = image.copy()
            draw = ImageDraw.Draw(highlighted_image)
            
            # Find contours of the mask
            mask_array = mask.astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_array, mode='L')
            
            # Resize mask to match image if needed (should not be needed with high-quality segmentation)
            if mask_img.size != image.size:
                logger.info(f"Resizing mask from {mask_img.size} to {image.size}")
                # Use LANCZOS for better quality instead of NEAREST
                mask_img = mask_img.resize(image.size, Image.LANCZOS)
                mask_array = np.array(mask_img)
                # Apply threshold to get clean binary mask
                mask_array = (mask_array > 128).astype(np.uint8) * 255
            
            # Apply advanced smoothing to eliminate blockiness
            try:
                from scipy import ndimage
                
                # Convert to float for better precision
                mask_float = mask_array.astype(float) / 255.0
                
                # Apply multiple smoothing passes for better quality
                # First pass: remove noise
                mask_smooth1 = ndimage.gaussian_filter(mask_float, sigma=0.8)
                
                # Second pass: smooth edges
                mask_smooth2 = ndimage.gaussian_filter(mask_smooth1, sigma=1.2)
                
                # Third pass: final smoothing
                mask_final = ndimage.gaussian_filter(mask_smooth2, sigma=0.6)
                
                # Apply threshold with hysteresis for cleaner edges
                mask_clean = (mask_final > 0.3).astype(np.uint8) * 255
                
                # Apply morphological operations for cleaner mask
                mask_clean = ndimage.binary_opening(mask_clean > 0, iterations=1)
                mask_clean = ndimage.binary_closing(mask_clean, iterations=1)
                
                mask_array = mask_clean.astype(np.uint8) * 255
                
                logger.info("Applied advanced smoothing and morphological operations for high-quality mask")
                
            except ImportError:
                logger.info("scipy not available, using basic smoothing")
                # Basic smoothing without scipy
                from PIL import ImageFilter
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.5))
                mask_array = np.array(mask_img)
                mask_array = (mask_array > 128).astype(np.uint8) * 255
            
            # Create outline by dilating and subtracting original mask
            try:
                from scipy import ndimage
                
                # Create smooth outline
                mask_bool = mask_array > 0
                
                # Dilate the mask to create outline
                dilated = ndimage.binary_dilation(mask_bool, iterations=2)
                outline = dilated & ~mask_bool
                
                logger.info(f"Outline created, outline pixels: {np.sum(outline)}")
                
                # Draw colored outline with anti-aliasing
                outline_coords = np.where(outline)
                if len(outline_coords[0]) > 0:
                    logger.info(f"Drawing {len(outline_coords[0])} outline pixels")
                    
                    # Color based on clothing type - now unified color for all
                    color = (34, 197, 94, 255)  # #22c55e for all clothing types
                    
                    # Draw smooth outline with anti-aliasing effect
                    for y, x in zip(outline_coords[0], outline_coords[1]):
                        if 0 <= y < highlighted_image.height and 0 <= x < highlighted_image.width:
                            # Create anti-aliasing effect with varying opacity
                            base_color = list(color)
                            base_color[3] = 255  # Full opacity for center
                            highlighted_image.putpixel((x, y), tuple(base_color))
                            
                            # Add semi-transparent pixels around for smoother edges
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    if dy == 0 and dx == 0:
                                        continue  # Skip center pixel
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < highlighted_image.height and 0 <= nx < highlighted_image.width:
                                        # Reduce opacity for edge pixels
                                        edge_color = list(color)
                                        edge_color[3] = 128  # 50% opacity
                                        highlighted_image.putpixel((nx, ny), tuple(edge_color))
                else:
                    logger.warning("No outline pixels found!")
                    # Fallback: create semi-transparent overlay
                    self._create_semi_transparent_overlay(highlighted_image, mask_array, selected_clothing)
                    
            except ImportError:
                logger.warning("scipy not available, using semi-transparent overlay method")
                # Create semi-transparent colored overlay
                self._create_semi_transparent_overlay(highlighted_image, mask_array, selected_clothing)
            
            # Convert to base64
            buffer = BytesIO()
            highlighted_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info("Highlight image created successfully")
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error creating highlighted image: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to original image
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    
    def _create_semi_transparent_overlay(self, image, mask_array, selected_clothing):
        """Create semi-transparent colored overlay for selected clothing."""
        try:
            # Unified color for all clothing types
            overlay_color = (34, 197, 94, 80)  # #22c55e with 30% transparency
            
            # Create overlay image
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_array = np.array(overlay)
            
            # Apply mask to overlay
            mask_bool = mask_array > 0
            overlay_array[mask_bool] = overlay_color
            
            # Convert back to PIL and composite
            overlay = Image.fromarray(overlay_array, 'RGBA')
            result = Image.alpha_composite(image, overlay)
            
            # Copy result back to original image
            image.paste(result, (0, 0))
            
            logger.info("Semi-transparent overlay created successfully with unified color")
            
        except Exception as e:
            logger.error(f"Error creating overlay: {e}")
            # If overlay fails, create simple colored border
            self._create_simple_border(image, mask_array, selected_clothing)
    
    def _create_simple_border(self, image, mask_array, selected_clothing):
        """Create simple colored border around detected clothing."""
        try:
            from PIL import ImageDraw
            
            # Find bounding box of the mask
            mask_bool = mask_array > 0
            if not np.any(mask_bool):
                logger.warning("No clothing detected for border creation")
                return
            
            # Get coordinates of clothing pixels
            coords = np.where(mask_bool)
            if len(coords[0]) == 0:
                return
            
            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])
            
            # Add padding around the bounding box
            padding = 5
            y_min = max(0, y_min - padding)
            y_max = min(image.height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(image.width - 1, x_max + padding)
            
            # Color based on clothing type - now unified color for all
            border_color = (34, 197, 94, 255)  # #22c55e for all clothing types
            
            # Draw border rectangle
            draw = ImageDraw.Draw(image)
            border_width = 3
            
            # Draw multiple rectangles for thicker border
            for i in range(border_width):
                draw.rectangle(
                    [x_min - i, y_min - i, x_max + i, y_max + i],
                    outline=border_color,
                    width=1
                )
            
            logger.info(f"Simple border created around clothing: ({x_min}, {y_min}) to ({x_max}, {y_max}) with unified color")
            
        except Exception as e:
            logger.error(f"Error creating simple border: {e}")
            # If everything fails, just return original image

    def process_multiple_images(self, image_bytes_list: list) -> list:
        """
        Process multiple images in batch for better efficiency.
        Returns list of results for each image.
        """
        try:
            results = []
            
            # Process images in parallel if possible
            import concurrent.futures
            from functools import partial
            
            # Use ThreadPoolExecutor for I/O bound operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all images for processing
                future_to_index = {
                    executor.submit(self.detect_clothing_with_segmentation, img_bytes): i 
                    for i, img_bytes in enumerate(image_bytes_list)
                }
                
                # Collect results in order
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results.append((index, result))
                    except Exception as e:
                        logger.error(f"Error processing image {index}: {e}")
                        results.append((index, {"error": str(e)}))
            
            # Sort results by original index
            results.sort(key=lambda x: x[0])
            return [result for _, result in results]
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Fallback to sequential processing
            return [self.detect_clothing_with_segmentation(img_bytes) for img_bytes in image_bytes_list]

def detect_clothing_types_with_segmentation(image_bytes: bytes) -> dict:
    """Get clothing detection with full segmentation data for reuse."""
    detector = get_clothing_detector()
    return detector.detect_clothing_with_segmentation(image_bytes)

def analyze_from_segmentation(segmentation_data: dict, selected_clothing: str = None) -> dict:
    """Analyze image using pre-computed segmentation data (much faster)."""
    detector = get_clothing_detector()
    return detector.analyze_from_segmentation(segmentation_data, selected_clothing)

# Global detector singleton (to reuse model)
_detector = None

def get_clothing_detector():
    """Get global detector instance (lazy-init)."""
    global _detector
    if _detector is None:
        _detector = ClothingDetector()
    return _detector

def detect_clothing_types(image_bytes: bytes) -> dict:
    """Convenience wrapper for clothing detection."""
    detector = get_clothing_detector()
    return detector.detect_clothing(image_bytes)

def create_clothing_only_image(image_bytes: bytes, selected_clothing: str = None) -> str:
    """Convenience wrapper for clothing-only image creation."""
    detector = get_clothing_detector()
    return detector.create_clothing_only_image(image_bytes, selected_clothing) 