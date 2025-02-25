"""
Weight pruning framework for LLM optimization.
Supports structured and unstructured pruning with configurable sparsity levels.
"""

import logging
from enum import Enum
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
from dataclasses import dataclass
from core.metrics import performance_monitor

logger = logging.getLogger(__name__)

class PruningMethod(Enum):
    MAGNITUDE = "magnitude"
    STRUCTURED = "structured"
    GRADIENT = "gradient"

@dataclass
class PruningConfig:
    """Configuration for pruning process."""
    method: PruningMethod
    target_sparsity: float
    granularity: str = "element"  # element, vector, or block
    schedule: str = "cubic"  # linear, cubic, or exponential
    block_size: Optional[Tuple[int, int]] = None
    min_threshold: float = 1e-3

class PruningOptimizer:
    def __init__(self, config: PruningConfig):
        """
        Initialize pruning optimizer.
        
        Args:
            config: Pruning configuration
        """
        self.config = config
        self.mask_dict: Dict[str, torch.Tensor] = {}
        self.stats: Dict[str, Dict] = {}
        logger.info(f"Initialized pruning optimizer with method: {config.method.value}")

    @performance_monitor
    def prune_model(
        self,
        model: torch.nn.Module,
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.nn.Module:
        """
        Prune model weights according to configuration.
        
        Args:
            model: PyTorch model to prune
            gradients: Optional gradients for gradient-based pruning
            
        Returns:
            Pruned model
        """
        logger.info(f"Starting model pruning with target sparsity: {self.config.target_sparsity}")
        
        try:
            # Apply pruning based on method
            if self.config.method == PruningMethod.MAGNITUDE:
                pruned_model = self._magnitude_pruning(model)
            elif self.config.method == PruningMethod.STRUCTURED:
                pruned_model = self._structured_pruning(model)
            else:  # GRADIENT method
                if gradients is None:
                    raise ValueError("Gradients required for gradient-based pruning")
                pruned_model = self._gradient_pruning(model, gradients)
            
            # Verify pruning results
            self._verify_pruning(pruned_model)
            
            return pruned_model
            
        except Exception as e:
            logger.error(f"Pruning failed: {str(e)}")
            raise

    @performance_monitor
    def _magnitude_pruning(self, model: torch.nn.Module) -> torch.nn.Module:
        """Implement magnitude-based pruning."""
        logger.debug("Applying magnitude-based pruning")
        
        for name, param in model.named_parameters():
            if param.dim() > 1:  # Only prune weight matrices
                # Calculate threshold
                abs_weights = param.abs()
                threshold = self._calculate_threshold(abs_weights)
                
                # Create and apply mask
                mask = (abs_weights > threshold).float()
                self.mask_dict[name] = mask
                
                # Apply pruning
                param.data.mul_(mask)
                
                # Update statistics
                self.stats[name] = {
                    'sparsity': 1.0 - (mask.sum() / mask.numel()),
                    'threshold': threshold
                }
        
        return model

    @performance_monitor
    def _structured_pruning(self, model: torch.nn.Module) -> torch.nn.Module:
        """Implement structured pruning."""
        logger.debug("Applying structured pruning")
        
        for name, param in model.named_parameters():
            if param.dim() > 1:
                if self.config.block_size:
                    # Block-wise pruning
                    mask = self._block_structured_pruning(param)
                else:
                    # Vector-wise pruning
                    mask = self._vector_structured_pruning(param)
                
                self.mask_dict[name] = mask
                param.data.mul_(mask)
                
                self.stats[name] = {
                    'sparsity': 1.0 - (mask.sum() / mask.numel())
                }
        
        return model

    @performance_monitor
    def _gradient_pruning(
        self,
        model: torch.nn.Module,
        gradients: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        """Implement gradient-based pruning."""
        logger.debug("Applying gradient-based pruning")
        
        for name, param in model.named_parameters():
            if name in gradients and param.dim() > 1:
                # Calculate importance scores based on gradients
                importance = self._calculate_importance(param, gradients[name])
                threshold = self._calculate_threshold(importance)
                
                # Create and apply mask
                mask = (importance > threshold).float()
                self.mask_dict[name] = mask
                param.data.mul_(mask)
                
                self.stats[name] = {
                    'sparsity': 1.0 - (mask.sum() / mask.numel()),
                    'threshold': threshold
                }
        
        return model

    def _block_structured_pruning(self, param: torch.Tensor) -> torch.Tensor:
        """Implement block-wise structured pruning."""
        if not self.config.block_size:
            raise ValueError("Block size must be specified for block structured pruning")
            
        h, w = self.config.block_size
        n_blocks_h = param.size(0) // h
        n_blocks_w = param.size(1) // w
        
        # Reshape into blocks
        blocks = param.view(n_blocks_h, h, n_blocks_w, w)
        block_norms = blocks.norm(dim=(1, 3))
        
        # Calculate threshold for block norms
        threshold = self._calculate_threshold(block_norms)
        
        # Create block mask
        block_mask = (block_norms > threshold).float()
        
        # Expand mask to original size
        mask = block_mask.repeat_interleave(h, dim=0).repeat_interleave(w, dim=1)
        
        return mask

    def _vector_structured_pruning(self, param: torch.Tensor) -> torch.Tensor:
        """Implement vector-wise structured pruning."""
        vector_norms = param.norm(dim=1)
        threshold = self._calculate_threshold(vector_norms)
        
        # Create vector mask
        vector_mask = (vector_norms > threshold).float()
        
        # Expand mask to match parameter shape
        mask = vector_mask.unsqueeze(1).expand_as(param)
        
        return mask

    def _calculate_threshold(self, values: torch.Tensor) -> float:
        """Calculate pruning threshold based on target sparsity."""
        k = int(values.numel() * (1 - self.config.target_sparsity))
        if k < 1:
            k = 1
        threshold = values.view(-1).kthvalue(k).values.item()
        return max(threshold, self.config.min_threshold)

    def _calculate_importance(
        self,
        param: torch.Tensor,
        gradient: torch.Tensor
    ) -> torch.Tensor:
        """Calculate importance scores for gradient-based pruning."""
        return param.abs() * gradient.abs()

    def _verify_pruning(self, model: torch.nn.Module):
        """Verify pruning results meet sparsity requirements."""
        total_params = 0
        total_nonzero = 0
        
        for name, param in model.named_parameters():
            if name in self.mask_dict:
                total_params += param.numel()
                total_nonzero += (param != 0).sum().item()
        
        actual_sparsity = 1.0 - (total_nonzero / total_params)
        logger.info(f"Achieved sparsity: {actual_sparsity:.4f}")
        
        if actual_sparsity < self.config.target_sparsity:
            logger.warning(
                f"Achieved sparsity ({actual_sparsity:.4f}) is less than "
                f"target ({self.config.target_sparsity:.4f})"
            )
