"""
Type definitions for streaming inference.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime

class TaskStatus(Enum):
    """Status of an inference task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""
    batch_size: int = 32
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream_interval: float = 0.1  # Seconds between token updates
    progress_interval: float = 1.0  # Seconds between progress updates
    timeout: float = 300.0  # Maximum runtime in seconds

@dataclass
class ProgressUpdate:
    """Progress update for inference task."""
    task_id: str
    status: TaskStatus
    total_tokens: int = 0
    generated_tokens: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None
        
    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self.total_tokens > 0:
            return min(100.0, (self.generated_tokens / self.total_tokens) * 100)
        return 0.0
        
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Get token generation speed."""
        duration = self.duration
        if duration and duration > 0:
            return self.generated_tokens / duration
        return None

@dataclass
class InferenceTask:
    """Streaming inference task."""
    task_id: str
    prompt: str
    config: StreamingConfig
    status: TaskStatus = TaskStatus.PENDING
    tokens: List[str] = field(default_factory=list)
    progress: Optional[ProgressUpdate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'tokens': self.tokens,
            'progress': {
                'status': self.progress.status.value,
                'progress': self.progress.progress,
                'tokens_per_second': self.progress.tokens_per_second,
                'duration': self.progress.duration,
                'error': self.progress.error
            } if self.progress else None,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceTask':
        """Create task from dictionary."""
        return cls(
            task_id=data['task_id'],
            prompt=data['prompt'],
            config=StreamingConfig(**data.get('config', {})),
            status=TaskStatus(data['status']),
            tokens=data.get('tokens', []),
            progress=ProgressUpdate(
                task_id=data['task_id'],
                status=TaskStatus(data['status']),
                **data.get('progress', {})
            ) if data.get('progress') else None,
            metadata=data.get('metadata', {})
        )
