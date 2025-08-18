"""
User management and identification for the Loomi Clothing Detection API.
"""
import hashlib
from typing import Optional
from fastapi import Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Security
security = HTTPBearer(auto_error=False)

def get_user_id(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """
    Extract user ID from request. 
    In production, validate JWT token.
    """
    if credentials:
        # In production, decode JWT and extract user_id
        return f"user_{hashlib.md5(credentials.credentials.encode()).hexdigest()[:8]}"
    
    # Fallback: use IP address + User-Agent hash
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    user_hash = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()[:8]
    return f"anon_{user_hash}"
