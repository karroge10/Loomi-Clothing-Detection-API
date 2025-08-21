---
title: Loomi Clothing Detection API
emoji: 🏢
colorFrom: indigo
colorTo: red
sdk: docker
sdk_version: "1.0.0"
app_file: main.py
pinned: false
---

# Loomi Clothing Detection API 🚀

AI-powered clothing analysis and segmentation API, optimized for Hugging Face Spaces.

## ⚠️ **Important Note for Hugging Face Demo**

**This is currently a demo version on Hugging Face Spaces with limitations:**
- **Only 1 request at a time** (Hugging Face Spaces restriction)
- **If you see "offline" status** → wait until current request completes
- **Try again in a few minutes** if the Space appears busy
- **For production use** → deploy to your own server with higher concurrency

## ✨ Features

- **🧠 AI-Powered**: Uses Segformer model for clothing detection
- **🖼️ Image Processing**: Background removal and dominant color detection
- **⚡ Fast**: Optimized for single-request processing with automatic caching
- **🔧 HF Optimized**: Built specifically for Hugging Face Spaces

## 🚀 Quick Start

### API Endpoints

- `GET /` - API overview and documentation
- `GET /health` - System health and status
- `GET /performance` - Performance statistics and cache info
- `POST /detect` - Detect clothing types with segmentation data
- `POST /analyze` - Upload same image for fast analysis using cached data

### Usage Example

```python
import requests

# Step 1: Upload image for clothing detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'https://your-hf-space.hf.space/detect',
        files={'file': f}
    )
    result = response.json()
    print(result)
    
# Step 2: Upload same image for instant analysis (uses cached data)
with open('image.jpg', 'rb') as f:
    analyze_response = requests.post(
        'https://your-hf-space.hf.space/analyze',
        files={'file': f},
        data={'selected_clothing': 'shirt'}  # Optional: specify clothing type
    )
    analysis = analyze_response.json()
    print(analysis)
```

## 🏗️ Architecture

- **FastAPI**: Modern, fast web framework
- **Efficient Processing**: Optimized for single requests with smart caching
- **Model Management**: Efficient ML model loading
- **Automatic Caching**: Smart caching for repeated images and segmentation data

## 🔧 Configuration

The API automatically detects Hugging Face Spaces and applies optimizations:

- Single worker process
- Optimized cache sizes
- HF-specific environment variables

## 📱 Integration

Perfect for:
- Mobile apps (React Native, Flutter)
- Web applications
- E-commerce platforms
- Fashion analysis tools

## 🚀 Running the API

```bash
# Simple startup
python run.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 7860
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 Models and Technologies

### **Core ML Model:**
- **Segformer B2 for Clothing Segmentation** - `mattmdjaga/segformer_b2_clothes`
  - Pre-trained model specifically designed for clothing and fashion item detection
  - Based on the Segformer architecture for semantic segmentation

### **Key Libraries and Licenses:**
- **Transformers (Hugging Face)** - Apache 2.0 License
- **PyTorch** - BSD License
- **Segformer** - MIT License  
- **Rembg** - MIT License (background removal)
- **FastAPI** - MIT License
- **Pillow (PIL)** - HPND License
- **NumPy** - BSD License
- **scikit-learn** - BSD License

### **Model Capabilities:**
The model can detect and segment 18 different categories:
- Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt
- Left/Right-shoe, Face, Left/Right-leg, Left/Right-arm, Bag, Scarf

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** While this project is MIT licensed, it uses pre-trained models and libraries with their own licenses. Please ensure compliance with all respective licenses when using this API.

---

**Made with ❤️ by the Loomi Team**

*AI-powered clothing analysis, simplified and ready! 🎯*
