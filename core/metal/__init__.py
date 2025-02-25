"""
Metal Performance Shaders (MPS) integration for optimized LLM operations on Apple Silicon.
Provides hardware-accelerated tensor operations and memory management for M4 Pro architecture.
"""

from .graph import MPSGraph
from .memory import UnifiedMemoryManager
from .kernels import MetalKernels

__all__ = ['MPSGraph', 'UnifiedMemoryManager', 'MetalKernels']
