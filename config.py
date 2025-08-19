import os
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class APIConfig:
    """Configuration class for the Loomi Clothing Detection API."""
    
    # API Settings
    title: str = "Loomi Clothing Detection API"
    description: str = "AI-powered clothing analysis and segmentation API with rate limiting and user management"
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
    
    # Rate Limiting
    rate_limit_requests: int = 10  # requests per minute
    rate_limit_window: int = 60  # seconds
    max_concurrent_requests: int = 5  # per user
    
    # Model Settings
    model_warmup_on_startup: bool = True
    model_cache_size: int = 10
    
    # Queue Settings
    num_workers: int = 2
    queue_max_size: int = 100
    
    # CORS Settings
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Security Settings
    enable_auth: bool = False
    jwt_secret: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    
    # Token System Settings (Future)
    enable_token_system: bool = False
    free_tier_requests_per_day: int = 100
    premium_tier_requests_per_day: int = 1000
    
    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring Settings
    enable_metrics: bool = False
    metrics_port: int = 8000
    
    # Cache Settings
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
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
        
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", str(self.rate_limit_requests)))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", str(self.rate_limit_window)))
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", str(self.max_concurrent_requests)))
        
        self.model_warmup_on_startup = os.getenv("MODEL_WARMUP_ON_STARTUP", str(self.model_warmup_on_startup)).lower() == "true"
        self.model_cache_size = int(os.getenv("MODEL_CACHE_SIZE", str(self.model_cache_size)))
        
        self.num_workers = int(os.getenv("NUM_WORKERS", str(self.num_workers)))
        self.queue_max_size = int(os.getenv("QUEUE_MAX_SIZE", str(self.queue_max_size)))
        
        # Handle allowed origins
        origins_env = os.getenv("ALLOWED_ORIGINS")
        if origins_env and origins_env != "*":
            self.allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
        
        self.enable_auth = os.getenv("ENABLE_AUTH", str(self.enable_auth)).lower() == "true"
        self.jwt_secret = os.getenv("JWT_SECRET", self.jwt_secret)
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", self.jwt_algorithm)
        self.jwt_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", str(self.jwt_expire_minutes)))
        
        self.enable_token_system = os.getenv("ENABLE_TOKEN_SYSTEM", str(self.enable_token_system)).lower() == "true"
        self.free_tier_requests_per_day = int(os.getenv("FREE_TIER_REQUESTS_PER_DAY", str(self.free_tier_requests_per_day)))
        self.premium_tier_requests_per_day = int(os.getenv("PREMIUM_TIER_REQUESTS_PER_DAY", str(self.premium_tier_requests_per_day)))
        
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.log_format = os.getenv("LOG_FORMAT", self.log_format)
        
        self.enable_metrics = os.getenv("ENABLE_METRICS", str(self.enable_metrics)).lower() == "true"
        self.metrics_port = int(os.getenv("METRICS_PORT", str(self.metrics_port)))
        
        self.enable_redis = os.getenv("ENABLE_REDIS", str(self.enable_redis)).lower() == "true"
        self.redis_host = os.getenv("REDIS_HOST", self.redis_host)
        self.redis_port = int(os.getenv("REDIS_PORT", str(self.redis_port)))
        self.redis_db = int(os.getenv("REDIS_DB", str(self.redis_db)))
        self.redis_password = os.getenv("REDIS_PASSWORD", self.redis_password)
        
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
        # Set HF environment variables (using modern HF_HOME instead of deprecated TRANSFORMERS_CACHE)
        os.environ["HF_HOME"] = self.hf_cache_dir
        # Note: TRANSFORMERS_CACHE is deprecated, using HF_HOME instead
        os.environ["HF_DATASETS_CACHE"] = f"{self.hf_cache_dir}/datasets"
        
        # Optimize for HF Spaces
        if self.workers > 1:
            self.workers = 1  # HF Spaces work better with single worker
        
        # Conservative rate limiting for HF
        if self.rate_limit_requests > 15:
            self.rate_limit_requests = 15
        
        if self.max_concurrent_requests > 5:
            self.max_concurrent_requests = 5
        
        # Smaller cache sizes for HF
        if self.model_cache_size > 5:
            self.model_cache_size = 5
        
        if self.queue_max_size > 25:
            self.queue_max_size = 25
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information for API responses."""
        return {
            "requests_per_minute": self.rate_limit_requests,
            "window_seconds": self.rate_limit_window,
            "concurrent_limit": self.max_concurrent_requests,
            "file_size_limit_mb": self.max_upload_mb
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for API responses."""
        return {
            "warmup_on_startup": self.model_warmup_on_startup,
            "cache_size": self.model_cache_size,
            "workers": self.num_workers
        }
    
    def get_security_info(self) -> Dict[str, Any]:
        """Get security information for API responses."""
        return {
            "authentication_enabled": self.enable_auth,
            "cors_enabled": True,
            "rate_limiting_enabled": True,
            "file_validation_enabled": True
        }
    
    def get_hf_info(self) -> Dict[str, Any]:
        """Get Hugging Face specific information."""
        return {
            "is_hf_space": self.is_huggingface_space,
            "space_id": self.space_id,
            "cache_dir": self.hf_cache_dir,
            "optimizations_applied": self.is_huggingface_space
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        if self.rate_limit_requests < 1:
            warnings.append("RATE_LIMIT_REQUESTS should be at least 1")
        
        if self.max_concurrent_requests < 1:
            warnings.append("MAX_CONCURRENT_REQUESTS should be at least 1")
        
        if self.max_upload_mb < 1:
            warnings.append("MAX_UPLOAD_MB should be at least 1")
        
        if self.workers < 1:
            warnings.append("WORKERS should be at least 1")
        
        if self.is_huggingface_space and self.workers > 1:
            warnings.append("Multiple workers not recommended in Hugging Face Spaces")
        
        if self.is_huggingface_space and self.rate_limit_requests > 20:
            warnings.append("High rate limits may cause issues in Hugging Face Spaces")
        
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
    print(f"  - Rate limit: {config.rate_limit_requests} requests per {config.rate_limit_window} seconds")
    print(f"  - Concurrent limit: {config.max_concurrent_requests} requests")
    print(f"  - File size limit: {config.max_upload_mb}MB")
    print(f"  - Workers: {config.workers}")
    print(f"  - Background workers: {config.num_workers}")
    print(f"  - Hugging Face Space: {config.is_huggingface_space}")
    if config.is_huggingface_space:
        print(f"  - Space ID: {config.space_id}")
        print(f"  - Cache dir: {config.hf_cache_dir}")
