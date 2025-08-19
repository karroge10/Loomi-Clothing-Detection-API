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

class ClothingDetector:
    def __init__(self):
        """Initialize clothing segmentation model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        # Note: feature_extractor_type and reduce_labels are deprecated in newer versions
        self.processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes",
            # Remove deprecated arguments that cause warnings
        )
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model.to(self.device)
        self.model.eval()
        
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
        
        logger.info("Clothing detector initialized successfully")
    
    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Create image hash to use as cache key."""
        return hashlib.md5(image_bytes).hexdigest()
    
    def _segment_image(self, image_bytes: bytes):
        """Run image segmentation with caching."""
        image_hash = self._get_image_hash(image_bytes)
        
        # Check cache
        if image_hash in _segmentation_cache:
            logger.info("Using cached segmentation result")
            return _segmentation_cache[image_hash]
        
        # Run segmentation
        logger.info("Performing new segmentation")
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu()
        
        # Upsample logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # Get predicted mask
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # Save to cache
        result = {
            'pred_seg': pred_seg,
            'image': image,
            'image_size': image.size
        }
        _segmentation_cache[image_hash] = result
        
        # Limit cache size (keep last 10)
        if len(_segmentation_cache) > 10:
            oldest_key = next(iter(_segmentation_cache))
            del _segmentation_cache[oldest_key]
        
        return result
    
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