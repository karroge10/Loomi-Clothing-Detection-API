"""
Rate limiting and user management for the Loomi Clothing Detection API.
"""
import asyncio
import time
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass
from config import config

@dataclass
class UserRequest:
    """Represents a user request for tracking."""
    timestamp: float
    endpoint: str
    file_size: int

class RateLimiter:
    """Manages rate limiting and concurrent request tracking per user."""
    
    def __init__(self):
        self.user_requests: Dict[str, List[UserRequest]] = defaultdict(list)
        self.user_concurrent: Dict[str, int] = defaultdict(int)
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """Check if user has exceeded rate limit."""
        async with self.lock:
            now = time.time()
            user_reqs = self.user_requests[user_id]
            
            # Remove old requests outside the window
            user_reqs = [req for req in user_reqs if now - req.timestamp < config.rate_limit_window]
            self.user_requests[user_id] = user_reqs
            
            # Check rate limit
            return len(user_reqs) < config.rate_limit_requests
    
    async def check_concurrent_limit(self, user_id: str) -> bool:
        """Check if user has exceeded concurrent request limit."""
        async with self.lock:
            return self.user_concurrent[user_id] < config.max_concurrent_requests
    
    async def add_request(self, user_id: str, endpoint: str, file_size: int):
        """Add a new request to user's history."""
        async with self.lock:
            now = time.time()
            self.user_requests[user_id].append(UserRequest(now, endpoint, file_size))
            self.user_concurrent[user_id] += 1
    
    async def remove_request(self, user_id: str):
        """Remove a completed request from concurrent count."""
        async with self.lock:
            if self.user_concurrent[user_id] > 0:
                self.user_concurrent[user_id] -= 1
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics for API responses."""
        now = time.time()
        user_reqs = self.user_requests[user_id]
        concurrent = self.user_concurrent[user_id]
        
        # Calculate usage in current window
        window_start = now - config.rate_limit_window
        requests_in_window = len([req for req in user_reqs if req.timestamp >= window_start])
        
        return {
            "user_id": user_id,
            "requests_in_window": requests_in_window,
            "requests_limit": config.rate_limit_requests,
            "concurrent_requests": concurrent,
            "concurrent_limit": config.max_concurrent_requests,
            "window_remaining": config.rate_limit_window - (now - window_start),
            "total_requests_today": len([req for req in user_reqs if req.timestamp >= now - 86400])
        }

# Global rate limiter instance
rate_limiter = RateLimiter()
