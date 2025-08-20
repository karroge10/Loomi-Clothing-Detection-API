import os
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class APIConfig:
    """Configuration class for the Loomi Clothing Detection API."""
    
    # API Settings
    title: str = "Loomi Clothing Detection API"
    description: str = "AI-powered clothing analysis and segmentation API"
    version: str = "2.0.0"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 7860
    workers: int = 1
    reload: bool = False
    
    # File Upload Settings
    max_upload_mb: int = 10
    max_upload_bytes: int = field(init=False)
    allowed_content_types: set = field(default_factory=lambda: {"image/jpeg", "image/png", "image/webp"})
    
    # Model Settings
    model_warmup_on_startup: bool = True
    model_cache_size: int = 10
    
    # CORS Settings
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Hugging Face Spaces Settings
    is_huggingface_space: bool = False
    space_id: str = ""
    hf_space: bool = False
    hf_cache_dir: str = "/tmp/hf_cache"
    
    def __post_init__(self):
        """Post-initialization to set computed fields and load from environment."""
        # Load from environment variables
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", str(self.port)))
        self.workers = int(os.getenv("WORKERS", str(self.workers)))
        self.reload = os.getenv("RELOAD", str(self.reload)).lower() == "true"
        
        self.max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", str(self.max_upload_mb)))
        self.max_upload_bytes = self.max_upload_mb * 1024 * 1024
        
        # Handle allowed content types
        content_types_env = os.getenv("ALLOWED_CONTENT_TYPES")
        if content_types_env:
            self.allowed_content_types = {c.strip() for c in content_types_env.split(",") if c.strip()}
        
        self.model_warmup_on_startup = os.getenv("MODEL_WARMUP_ON_STARTUP", str(self.model_warmup_on_startup)).lower() == "true"
        self.model_cache_size = int(os.getenv("MODEL_CACHE_SIZE", str(self.model_cache_size)))
        
        # Handle allowed origins
        origins_env = os.getenv("ALLOWED_ORIGINS")
        if origins_env and origins_env != "*":
            self.allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
        
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.log_format = os.getenv("LOG_FORMAT", self.log_format)
        
        # Hugging Face detection and settings
        self.space_id = os.getenv("SPACE_ID", "")
        self.hf_space = os.getenv("HF_SPACE", str(self.hf_space)).lower() == "true"
        self.hf_cache_dir = os.getenv("HF_CACHE_DIR", self.hf_cache_dir)
        
        # Determine if this is a Hugging Face Space
        self.is_huggingface_space = bool(self.space_id.strip()) or self.hf_space
        
        # Apply HF-specific optimizations
        if self.is_huggingface_space:
            self._apply_hf_optimizations()
    
    def _apply_hf_optimizations(self):
        """Apply Hugging Face Spaces specific optimizations."""
        # Set HF environment variables
        os.environ["HF_HOME"] = self.hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = f"{self.hf_cache_dir}/datasets"
        
        # CPU optimizations for free tier
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        
        # Optimize for HF Spaces
        if self.workers > 1:
            self.workers = 1  # HF Spaces work better with single worker
        
        # Smaller cache sizes for HF
        if self.model_cache_size > 5:
            self.model_cache_size = 5
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        if self.max_upload_mb < 1:
            warnings.append("MAX_UPLOAD_MB should be at least 1")
        
        if self.workers < 1:
            warnings.append("WORKERS should be at least 1")
        
        if self.is_huggingface_space and self.workers > 1:
            warnings.append("Multiple workers not recommended in Hugging Face Spaces")
        
        return warnings

# Global configuration instance
config = APIConfig()

# Validate configuration on import
if __name__ == "__main__":
    warnings = config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Configuration is valid!")
    
    print(f"\nCurrent configuration:")
    print(f"  - File size limit: {config.max_upload_mb}MB")
    print(f"  - Workers: {config.workers}")
    print(f"  - Hugging Face Space: {config.is_huggingface_space}")
    if config.is_huggingface_space:
        print(f"  - Space ID: {config.space_id}")
        print(f"  - Cache dir: {config.hf_cache_dir}")
