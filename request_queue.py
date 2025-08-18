"""
Request queue management for the Loomi Clothing Detection API.
"""
import asyncio
import logging
from typing import Any, Callable
from config import config

logger = logging.getLogger(__name__)

class RequestQueue:
    """Manages background processing of heavy API requests."""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.processing = False
        self.workers = []
    
    async def start_workers(self, num_workers: int = None):
        """Start background workers."""
        if num_workers is None:
            num_workers = config.num_workers
            
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        logger.info(f"Started {num_workers} background workers")
    
    async def _worker(self, name: str):
        """Background worker for processing requests."""
        logger.info(f"Worker {name} started")
        while True:
            try:
                task = await self.queue.get()
                if task is None:  # Shutdown signal
                    break
                
                user_id, endpoint, process_func, args, future = task
                try:
                    # Process the request
                    result = await process_func(*args)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker {name} error: {e}")
    
    async def submit_task(self, user_id: str, endpoint: str, process_func: Callable, *args) -> Any:
        """Submit a task to the queue."""
        future = asyncio.Future()
        await self.queue.put((user_id, endpoint, process_func, args, future))
        return await future
    
    async def shutdown(self):
        """Shutdown workers."""
        for _ in self.workers:
            await self.queue.put(None)
        await asyncio.gather(*self.workers, return_exceptions=True)
    
    def get_status(self) -> dict:
        """Get current queue status for health checks."""
        return {
            "queue_size": self.queue.qsize(),
            "active_workers": len(self.workers)
        }

# Global request queue instance
request_queue = RequestQueue()
