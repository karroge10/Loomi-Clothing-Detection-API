---
title: Loomi Clothing Detection API
emoji: 🏢
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: Clothing segmentation and background removal
---

# Loomi Clothing Detection API 🚀

AI-powered clothing analysis and segmentation API, optimized for Hugging Face Spaces.

## ✨ Features

- **🧠 AI-Powered**: Uses Segformer model for clothing detection
- **🖼️ Image Processing**: Background removal and dominant color detection
- **⚡ Async**: Non-blocking model loading and request processing
- **🚦 Rate Limiting**: Per-user request limits and concurrent control
- **👥 Multi-User**: Supports multiple users with isolation
- **🔧 HF Optimized**: Built specifically for Hugging Face Spaces

## 🚀 Quick Start

### API Endpoints

- `GET /` - API overview
- `GET /health` - System health and status
- `GET /user/stats` - User usage statistics
- `POST /clothing` - Detect clothing types and coordinates
- `POST /analyze` - Full analysis with color detection
- `POST /analyze/download` - Download processed images

### Usage Example

```python
import requests

# Upload image for clothing detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'https://your-hf-space.hf.space/clothing',
        files={'file': f}
    )
    result = response.json()
    print(result)
```

## 🏗️ Architecture

- **FastAPI**: Modern, fast web framework
- **Async Processing**: Non-blocking operations
- **Rate Limiting**: User-based request control
- **Model Management**: Efficient ML model loading
- **Queue System**: Background task processing

## 🔧 Configuration

The API automatically detects Hugging Face Spaces and applies optimizations:

- Single worker process
- Conservative rate limits (15 req/min, 5 concurrent)
- Optimized cache sizes
- HF-specific environment variables

## 📱 Integration

Perfect for:
- Mobile apps (React Native, Flutter)
- Web applications
- E-commerce platforms
- Fashion analysis tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Made with ❤️ by the Loomi Team**

*AI-powered clothing analysis, production ready! 🎯*
