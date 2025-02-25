"""Security middleware module."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
from typing import Optional
from .rate_limit import RateLimiter
import logging

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for FastAPI applications."""
    
    def __init__(
        self,
        app: ASGIApp,
        csrf_enabled: bool = True,
        rate_limit_enabled: bool = True
    ):
        super().__init__(app)
        self.csrf_enabled = csrf_enabled
        self.rate_limit_enabled = rate_limit_enabled
    
    async def dispatch(
        self,
        request: Request,
        call_next
    ) -> Response:
        """Process the request through security middleware."""
        start_time = time.time()
        
        try:
            # Rate limiting
            if self.rate_limit_enabled:
                # Determine rate limit type based on endpoint
                limit_type = "default"
                if request.url.path.startswith("/auth"):
                    limit_type = "auth"
                elif request.url.path.startswith("/inference"):
                    limit_type = "inference"
                
                await RateLimiter.check_rate_limit(request, limit_type)
            
            # CSRF protection for state-changing operations
            if self.csrf_enabled and request.method in ["POST", "PUT", "DELETE", "PATCH"]:
                self._verify_csrf(request)
            
            # Process the request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'"
            )
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            # Add request ID for tracking
            request_id = self._get_request_id(request)
            if request_id:
                response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}", exc_info=True)
            raise
        finally:
            # Log request details
            duration = time.time() - start_time
            self._log_request(request, duration)
    
    def _verify_csrf(self, request: Request) -> None:
        """Verify CSRF token."""
        token = request.headers.get("X-CSRF-Token")
        if not token:
            raise Exception("CSRF token missing")
        # Add your CSRF token validation logic here
    
    def _get_request_id(self, request: Request) -> Optional[str]:
        """Get or generate request ID."""
        return request.headers.get("X-Request-ID")
    
    def _log_request(self, request: Request, duration: float) -> None:
        """Log request details."""
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"Client: {request.client.host} "
            f"Duration: {duration:.3f}s"
        )
