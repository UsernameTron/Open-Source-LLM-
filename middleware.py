import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from config import settings

# Simple in-memory rate limiter
class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, Tuple[int, float]] = {}
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        if client_ip in self.requests:
            requests, window_start = self.requests[client_ip]
            time_passed = now - window_start
            
            # Reset window if more than a minute has passed
            if time_passed > 60:
                self.requests[client_ip] = (1, now)
                return True
            
            # Check if rate limit is exceeded
            if requests >= settings.RATE_LIMIT_PER_MINUTE:
                return False
            
            # Increment request count
            self.requests[client_ip] = (requests + 1, window_start)
            return True
        
        # First request from this IP
        self.requests[client_ip] = (1, now)
        return True

rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    # Continue processing the request
    response = await call_next(request)
    return response

async def verify_api_key_middleware(request: Request, call_next):
    if settings.API_KEY:  # Only check if API_KEY is configured
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if not api_key or api_key != settings.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )
    
    response = await call_next(request)
    return response
