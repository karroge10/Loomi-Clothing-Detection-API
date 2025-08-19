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

# Suppress transformers warnings for cleaner logs
warnings.filterwarnings("ignore", message=".*feature_extractor_type.*")
warnings.filterwarnings("ignore", message=".*reduce_labels.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for segmentation results
_segmentation_cache = {}
_cache_hits = 0
_cache_misses = 0

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
        
        # Check cache
        if image_hash in _segmentation_cache:
            global _cache_hits
            _cache_hits += 1
            logger.info("Using cached segmentation result")
            return _segmentation_cache[image_hash]
        
        global _cache_misses
        _cache_misses += 1
        # Run segmentation
        logger.info("Performing new segmentation")
        
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
            
            # Store result in cache
            _segmentation_cache[image_hash] = {
                'pred_seg': pred_seg,
                'image': image
            }
            
            # Limit cache size (keep last 10)
            if len(_segmentation_cache) > 10:
                oldest_key = next(iter(_segmentation_cache))
                del _segmentation_cache[oldest_key]
            
            return {
                'pred_seg': pred_seg,
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
            # Get cached segmentation result
            seg_result = self._segment_image(image_bytes)
            pred_seg = seg_result['pred_seg']
            image = seg_result['image']
            
            # Get clothing detection
            clothing_result = self.detect_clothing(image_bytes)
            
            # Convert original image to base64 for reuse
            import base64
            from io import BytesIO
            
            # Save original image as base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            original_image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Return both clothing info and segmentation data
            return {
                **clothing_result,
                "segmentation_data": {
                    "pred_seg": pred_seg.tolist(),  # Convert numpy array to list for JSON
                    "image_size": image.size,
                    "image_hash": self._get_image_hash(image_bytes),
                    "original_image": f"data:image/png;base64,{original_image_base64}"  # Add original image
                }
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
            
            # Convert mask to PIL image
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
            
            # Resize mask to match original image if needed
            if mask_img.size != original_image.size:
                mask_img = mask_img.resize(original_image.size, Image.NEAREST)
            
            # Convert original image to RGBA if it's not already
            if original_image.mode != 'RGBA':
                original_image = original_image.convert('RGBA')
            
            # Create new image with transparent background
            result = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
            
            # Apply mask to original image
            # Copy pixels where mask is white (255), make transparent where mask is black (0)
            original_array = np.array(original_image)
            mask_array = np.array(mask_img)
            
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