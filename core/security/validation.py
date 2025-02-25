"""Input validation module."""

from typing import Optional, List
from pydantic import BaseModel, validator, constr
import re

class SecureInput(BaseModel):
    """Base model for secure input validation."""
    text: constr(max_length=1000)
    metadata: Optional[dict] = None
    
    @validator('text')
    def validate_text(cls, v: str) -> str:
        """Validate input text for security concerns."""
        # Check for potential XSS/injection patterns
        dangerous_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'onerror=',
            r'onload=',
            r'eval\(',
            r'document\.',
            r'window\.',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially dangerous input detected')
        
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate metadata for security concerns."""
        if v is None:
            return v
            
        # Limit metadata size
        if len(str(v)) > 5000:
            raise ValueError('Metadata too large')
            
        # Check for nested depth
        def check_depth(obj, depth=0):
            if depth > 5:  # Maximum nesting depth
                raise ValueError('Metadata nesting too deep')
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, depth + 1)
        
        check_depth(v)
        return v

class BatchInput(SecureInput):
    """Model for batch input validation."""
    batch_size: int
    priority: Optional[int] = 1
    
    @validator('batch_size')
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if v < 1:
            raise ValueError('Batch size must be positive')
        if v > 128:
            raise ValueError('Batch size too large')
        return v
    
    @validator('priority')
    def validate_priority(cls, v: Optional[int]) -> Optional[int]:
        """Validate priority."""
        if v is not None and not (0 <= v <= 10):
            raise ValueError('Priority must be between 0 and 10')
        return v

def sanitize_file_path(path: str) -> str:
    """Sanitize file paths to prevent path traversal attacks."""
    # Remove any parent directory references
    path = re.sub(r'\.\./', '', path)
    path = re.sub(r'\.\.\\', '', path)
    
    # Remove any double slashes
    path = re.sub(r'/+', '/', path)
    
    # Remove any non-standard characters
    path = re.sub(r'[^a-zA-Z0-9./\-_]', '', path)
    
    return path
