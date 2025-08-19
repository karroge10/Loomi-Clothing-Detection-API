# Loomi Clothing Detection API - Integration Instructions

## API Overview

The Loomi Clothing Detection API is an AI-powered service that provides clothing detection, segmentation, and analysis capabilities. It's designed with a two-step workflow for optimal performance and user experience.

## Base URL
```
http://localhost:8000  # Local development
https://huggingface.co/spaces/karoge/Loomi-Clothing-Detection-API  # Production
```

## API Endpoints

### 1. Health Check
```
GET /health
```
Returns API status and basic information.

### 2. Performance Statistics
```
GET /performance
```
Returns cache statistics, device info, and performance metrics.

### 3. Main Detection Endpoint
```
POST /detect
```
**Purpose**: First step - detect clothing types and get segmentation data
**Content-Type**: `multipart/form-data`
**Body**: `file` (image file)

**Response**:
```json
{
  "clothing_types": [
    {
      "type": "shirt",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 300],
      "mask": "base64_encoded_mask_data"
    }
  ],
  "segmentation_data": {
    "masks": {
      "shirt": "base64_encoded_mask",
      "pants": "base64_encoded_mask"
    },
    "image_size": [800, 600]
  },
  "processing_time": 2.5
}
```

### 4. Analysis Endpoint
```
POST /analyze
```
**Purpose**: Second step - analyze selected clothing using segmentation data
**Content-Type**: `application/json`
**Body**:
```json
{
  "segmentation_data": {
    "masks": {...},
    "image_size": [800, 600]
  },
  "selected_clothing": "shirt"
}
```

**Response**:
```json
{
  "selected_clothing": "shirt",
  "background_removed": "base64_encoded_image",
  "dominant_colors": [
    {
      "color": [255, 128, 0],
      "percentage": 45.2,
      "hex": "#ff8000"
    }
  ],
  "processing_time": 1.2
}
```

## User Workflow

### Step 1: Image Upload & Detection
1. User uploads an image through your frontend
2. Frontend sends image to `/detect` endpoint
3. API processes image and returns:
   - List of detected clothing types
   - Segmentation masks for each type
   - Processing time
4. Frontend displays detected clothing types with confidence scores

### Step 2: Clothing Selection & Analysis
1. User selects a specific clothing item from the detected types
2. Frontend sends segmentation data + selected clothing to `/analyze` endpoint
3. API processes the selected clothing and returns:
   - Background-removed image
   - Dominant color analysis
   - Processing time
4. Frontend displays the analyzed results

## Visual Display & Highlighting

### Clothing Type Highlighting
When clothing types are detected, each item should be visually highlighted on the image:

- **Default State**: All detected clothing items are highlighted with a **unified green color** (#22c55e)
- **Selected State**: When user selects a specific clothing type, it gets highlighted with **enhanced visibility**
- **Color Scheme**: 
  - Primary highlight: `#22c55e` (green) with 100% opacity
  - Edge smoothing: `#22c55e` with 50% opacity for anti-aliasing effect
  - Overlay: `#22c55e` with 30% transparency for background overlay

### What to Display

#### 1. Detection Results View
- **Original image** with all detected clothing items highlighted
- **Clothing type list** showing:
  - Type name (shirt, pants, dress, etc.)
  - Confidence score (0.0 - 1.0)
  - Bounding box coordinates
- **Interactive selection** - clicking on clothing type or image area selects that item

#### 2. Analysis Results View
- **Background-removed image** of the selected clothing item
- **Dominant color palette** with:
  - Color swatches
  - Percentage distribution
  - Hex color codes
- **Processing statistics** (time taken, model confidence)

#### 3. Interactive Elements
- **Clickable clothing types** in the list
- **Image click detection** to select clothing items
- **Hover effects** to preview selection
- **Reset button** to analyze new images

### Highlighting Implementation

The API provides segmentation masks that you can use to create visual overlays:

```javascript
// Example: Create highlighted image with selected clothing
const createHighlightedImage = (originalImage, segmentationMask, selectedClothing) => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Draw original image
  ctx.drawImage(originalImage, 0, 0);
  
  // Apply green highlight overlay using the mask
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const maskData = decodeBase64Mask(segmentationMask);
  
  for (let i = 0; i < imageData.data.length; i += 4) {
    if (maskData[i / 4] > 0) {
      // Apply green highlight with transparency
      imageData.data[i] = 34;     // R: 34
      imageData.data[i + 1] = 197; // G: 197  
      imageData.data[i + 2] = 94;  // B: 94
      imageData.data[i + 3] = 128; // A: 50% opacity
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
};
```

## Integration Code Examples

### JavaScript/TypeScript

#### Step 1: Detect Clothing
```javascript
const detectClothing = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  try {
    const response = await fetch('http://localhost:8000/detect', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Detection failed:', error);
    throw error;
  }
};
```

#### Step 2: Analyze Selected Clothing
```javascript
const analyzeClothing = async (segmentationData, selectedClothing) => {
  try {
    const response = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        segmentation_data: segmentationData,
        selected_clothing: selectedClothing,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Analysis failed:', error);
    throw error;
  }
};
```

### React Hook Example
```javascript
import { useState, useCallback } from 'react';

export const useClothingAPI = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  const detectClothing = useCallback(async (imageFile) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await detectClothing(imageFile);
      setDetectionResult(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const analyzeClothing = useCallback(async (segmentationData, selectedClothing) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await analyzeClothing(segmentationData, selectedClothing);
      setAnalysisResult(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    detectClothing,
    analyzeClothing,
    detectionResult,
    analysisResult,
    isLoading,
    error,
  };
};
```

## Error Handling

### Rate Limiting (429)
```json
{
  "detail": "Rate limit exceeded. Maximum 15 requests per 60 seconds."
}
```
**Solution**: Implement exponential backoff and show retry countdown

### Concurrent Limit (429)
```json
{
  "detail": "Concurrent request limit exceeded. Maximum 5 concurrent requests."
}
```
**Solution**: Queue requests and show "processing" status

### File Validation (400)
```json
{
  "detail": "File must be an image"
}
```
**Solution**: Validate file type before upload

### Server Error (500)
```json
{
  "error": "Internal server error",
  "detail": "Error description",
  "type": "ErrorType"
}
```
**Solution**: Show user-friendly error message and retry option

## Rate Limiting Details

- **Requests per window**: 15 requests per 60 seconds
- **Concurrent requests**: Maximum 5 concurrent requests per user
- **Headers returned**:
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Time when rate limit resets

## Best Practices

### 1. Image Optimization
- Compress images before upload (max 10MB)
- Use common formats: JPEG, PNG, WebP
- Recommended dimensions: 800x600 to 1920x1080

### 2. User Experience
- Show loading states during API calls
- Implement retry mechanisms for failed requests
- Cache segmentation data to avoid re-detection
- Display progress indicators for long operations

### 3. Error Handling
- Implement graceful fallbacks
- Show user-friendly error messages
- Provide retry options for recoverable errors
- Log errors for debugging

### 4. Performance
- Use the two-step workflow (detect → analyze)
- Cache segmentation results
- Implement request queuing for concurrent limits
- Show estimated processing times

## Complete Integration Flow

```javascript
// 1. User selects image
const handleImageUpload = async (file) => {
  try {
    // Step 1: Detect clothing
    const detectionResult = await detectClothing(file);
    
    // Display detected clothing types
    displayClothingTypes(detectionResult.clothing_types);
    
    // Create highlighted image with all detected items
    const highlightedImage = createHighlightedImage(
      file, 
      detectionResult.segmentation_data.masks
    );
    setHighlightedImage(highlightedImage);
    
    // Store segmentation data for later use
    setSegmentationData(detectionResult.segmentation_data);
    
  } catch (error) {
    handleError(error);
  }
};

// 2. User selects clothing type
const handleClothingSelect = async (clothingType) => {
  try {
    // Update visual highlighting for selected item
    const selectedHighlightedImage = createSelectedHighlightedImage(
      highlightedImage, 
      segmentationData.masks[clothingType]
    );
    setSelectedHighlightedImage(selectedHighlightedImage);
    
    // Step 2: Analyze selected clothing
    const analysisResult = await analyzeClothing(
      segmentationData, 
      clothingType
    );
    
    // Display analysis results
    displayAnalysisResults(analysisResult);
    
  } catch (error) {
    handleError(error);
  }
};

// Helper function to create highlighted image for all detected items
const createHighlightedImage = (imageFile, masks) => {
  // Implementation using canvas to overlay green highlights
  // on all detected clothing areas
};

// Helper function to create enhanced highlighting for selected item
const createSelectedHighlightedImage = (baseImage, selectedMask) => {
  // Implementation to show selected clothing with enhanced visibility
  // while keeping other items with standard highlighting
};
```

## Visual Implementation Details

### 1. Image Highlighting States

#### Initial Detection State
- Load original image
- Apply green overlay (#22c55e) to all detected clothing areas
- Use 30% transparency for subtle highlighting
- Show bounding boxes around each detected item

#### Selection State  
- Keep all items highlighted with standard green
- Enhance selected item with:
  - Brighter green outline (#22c55e with 100% opacity)
  - Anti-aliased edges for smooth appearance
  - Slightly thicker border for emphasis

### 2. Interactive Elements

#### Clickable Areas
- **Image clicks**: Detect if click is within clothing mask bounds
- **List clicks**: Select clothing type from sidebar
- **Hover effects**: Preview selection before clicking

#### Visual Feedback
- **Hover state**: Slightly brighter highlighting
- **Selected state**: Enhanced border and outline
- **Processing state**: Loading spinner over selected area

### 3. Mask Processing

**Base64 Mask Decoding:**
- API returns base64-encoded segmentation masks
- Use `atob()` to decode base64 to binary string
- Convert to Uint8Array for pixel processing

**Mask Overlay Application:**
- Iterate through image pixels (RGBA format)
- Apply green highlight color (#22c55e) where mask > 0
- Use transparency levels for different visual effects

## Testing

### Test Images
- Use images with clear clothing items
- Test with different clothing types (shirts, pants, dresses)
- Test with various backgrounds and lighting conditions

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test detection endpoint
curl -X POST -F "file=@test-image.jpg" http://localhost:8000/detect

# Test analysis endpoint
curl -X POST -H "Content-Type: application/json" \
  -d '{"segmentation_data": {...}, "selected_clothing": "shirt"}' \
  http://localhost:8000/analyze
```

## Support & Troubleshooting

- Check API health endpoint for system status
- Monitor performance endpoint for cache statistics
- Implement proper logging for debugging
- Use browser dev tools to inspect API responses
- Check rate limiting headers for usage information

This API is designed to be simple yet powerful, providing fast clothing detection and analysis while maintaining good performance through intelligent caching and optimization.

## UI Component Examples

### Component Logic Description

**Clothing Type List Component:**
- Renders list of detected clothing types
- Each item shows: type name, confidence percentage, bounding box coordinates
- Clicking item calls `onSelect(item.type)` function
- Selected item gets visual "selected" state

**Highlighted Image Component:**
- Displays image with clothing highlighting
- Handles click events to detect clothing selection
- Shows processing overlay with spinner when analyzing
- Click coordinates are passed to `onClothingClick(x, y)` function

### Visual States Description

**Clothing Item States:**
- **Default**: Transparent background with subtle border
- **Hover**: Light green background with green border (rgba(34, 197, 94, 0.1) background, rgba(34, 197, 94, 0.3) border)
- **Selected**: Medium green background with solid green border (#22c55e) and green shadow

**Image States:**
- **Detection Image**: Crosshair cursor to indicate clickable areas
- **Processing Overlay**: Semi-transparent dark overlay with centered spinner and "Analyzing..." text

## Summary

The Loomi Clothing Detection API provides a complete solution for clothing analysis with:

1. **Two-step workflow** for optimal performance
2. **Visual highlighting** with unified green color scheme (#22c55e)
3. **Interactive selection** through clicks and hover effects
4. **Real-time feedback** with loading states and progress indicators
5. **Comprehensive error handling** for rate limiting and validation

The visual system uses consistent green highlighting (#22c55e) with varying opacity levels to create a professional and intuitive user experience that clearly shows detected clothing items and selected elements.
