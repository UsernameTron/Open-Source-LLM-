"""Rate limiting module."""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Optional
from fastapi import HTTPException, Request
import asyncio
from datetime import datetime, timedelta

@dataclass
class RateLimit:
    """Rate limit configuration."""
    max_requests: int
    window_seconds: int

class RateLimiter:
    """Rate limiter implementation using sliding window."""
    
    # Store request timestamps for each client
    _requests: DefaultDict[str, list] = defaultdict(list)
    _locks: Dict[str, asyncio.Lock] = {}
    
    # Default rate limits
    DEFAULT_LIMITS = {
        "default": RateLimit(100, 60),  # 100 requests per minute
        "auth": RateLimit(20, 60),      # 20 auth requests per minute
        "inference": RateLimit(50, 60),  # 50 inference requests per minute
    }
    
    @classmethod
    async def get_lock(cls, key: str) -> asyncio.Lock:
        """Get or create a lock for a given key."""
        if key not in cls._locks:
            cls._locks[key] = asyncio.Lock()
        return cls._locks[key]
    
    @classmethod
    def _clean_old_requests(cls, client_id: str, window: int) -> None:
        """Remove requests outside the current window."""
        current_time = time.time()
        cls._requests[client_id] = [
            req_time for req_time in cls._requests[client_id]
            if current_time - req_time <= window
        ]
    
    @classmethod
    async def check_rate_limit(
        cls,
        request: Request,
        limit_type: str = "default"
    ) -> None:
        """Check if request is within rate limits."""
        # Get client identifier (IP address or API key)
        client_id = request.client.host
        if api_key := request.headers.get("X-API-Key"):
            client_id = f"{client_id}:{api_key}"
        
        # Get rate limit configuration
        rate_limit = cls.DEFAULT_LIMITS.get(limit_type, cls.DEFAULT_LIMITS["default"])
        
        # Get lock for this client
        async with await cls.get_lock(client_id):
            # Clean old requests
            cls._clean_old_requests(client_id, rate_limit.window_seconds)
            
            # Check current request count
            current_count = len(cls._requests[client_id])
            
            if current_count >= rate_limit.max_requests:
                # Calculate reset time
                oldest_request = cls._requests[client_id][0]
                reset_time = oldest_request + rate_limit.window_seconds
                wait_seconds = int(reset_time - time.time())
                
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "reset_in_seconds": wait_seconds,
                        "limit": rate_limit.max_requests,
                        "window_seconds": rate_limit.window_seconds
                    }
                )
            
            # Add current request
            cls._requests[client_id].append(time.time())
    
    @classmethod
    def update_rate_limit(cls, limit_type: str, max_requests: int, window_seconds: int) -> None:
        """Update rate limit configuration."""
        cls.DEFAULT_LIMITS[limit_type] = RateLimit(max_requests, window_seconds)
