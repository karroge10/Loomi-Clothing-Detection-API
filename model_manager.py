"""
Async model management for the Loomi Clothing Detection API.
"""
import asyncio
import logging
from config import config

logger = logging.getLogger(__name__)

class AsyncModelManager:
    """Manages asynchronous loading of the ML model."""
    
    def __init__(self):
        self.model_loaded = False
        self.model_loading = False
        self.loading_task = None
        self._lock = asyncio.Lock()
    
    async def ensure_model_loaded(self):
        """Ensure model is loaded, load it asynchronously if needed."""
        if self.model_loaded:
            return
        
        async with self._lock:
            if self.model_loaded:
                return
            
            if self.model_loading:
                # Wait for existing loading task
                if self.loading_task:
                    await self.loading_task
                return
            
            # Start loading task
            self.model_loading = True
            self.loading_task = asyncio.create_task(self._load_model())
            await self.loading_task
    
    async def _load_model(self):
        """Load model in background thread."""
        try:
            logger.info("Starting model loading in background...")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            
            self.model_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loading = False
        finally:
            self.model_loading = False
    
    def _load_model_sync(self):
        """Synchronous model loading (runs in thread pool)."""
        try:
            from clothing_detector import get_clothing_detector
            detector = get_clothing_detector()
            logger.info("Model loaded in background thread")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_status(self) -> dict:
        """Get current model status for health checks."""
        return {
            "model_loaded": self.model_loaded,
            "model_loading": self.model_loading
        }

# Global model manager instance
model_manager = AsyncModelManager()
