"""
Attention map visualization for transformer models.
Provides real-time visualization of attention patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from core.metrics import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """Configuration for attention visualization."""
    layer_indices: Optional[List[int]] = None
    head_indices: Optional[List[int]] = None
    max_tokens: int = 100
    update_interval_ms: int = 100
    colorscale: str = 'Viridis'
    interpolation: str = 'best'

class AttentionVisualizer:
    def __init__(self, config: Optional[AttentionConfig] = None):
        """
        Initialize attention visualizer.
        
        Args:
            config: Optional visualization configuration
        """
        self.config = config or AttentionConfig()
        self.figure = None
        self._setup_visualization()
        logger.info("Initialized attention visualizer")

    @performance_monitor
    def _setup_visualization(self):
        """Setup the visualization layout."""
        self.figure = go.Figure()
        self.figure.update_layout(
            title="Attention Patterns",
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            template="plotly_dark",
            showlegend=True,
            uirevision=True  # Preserve zoom level on updates
        )

    @performance_monitor
    def visualize_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> go.Figure:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weight tensor [batch, heads, query_len, key_len]
            tokens: List of token strings
            layer_idx: Optional layer index to visualize
            head_idx: Optional attention head index to visualize
            
        Returns:
            Plotly figure object
        """
        try:
            # Process attention weights
            weights = self._process_attention_weights(attention_weights, layer_idx, head_idx)
            
            # Truncate tokens if needed
            tokens = self._truncate_tokens(tokens)
            
            # Create heatmap
            self._update_heatmap(weights, tokens)
            
            return self.figure
            
        except Exception as e:
            logger.error(f"Error visualizing attention: {str(e)}")
            raise

    def _process_attention_weights(
        self,
        weights: torch.Tensor,
        layer_idx: Optional[int],
        head_idx: Optional[int]
    ) -> np.ndarray:
        """Process attention weights for visualization."""
        # Convert to numpy and detach from computation graph
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        # Select specific layer/head if specified
        if layer_idx is not None:
            weights = weights[layer_idx]
        if head_idx is not None:
            weights = weights[head_idx]
        
        # Ensure proper shape
        if weights.ndim == 4:  # [batch, heads, query_len, key_len]
            weights = weights[0]  # Take first batch
        if weights.ndim == 3:  # [heads, query_len, key_len]
            weights = weights.mean(axis=0)  # Average over heads
            
        return weights

    def _truncate_tokens(self, tokens: List[str]) -> List[str]:
        """Truncate token list if needed."""
        if len(tokens) > self.config.max_tokens:
            logger.warning(f"Truncating tokens from {len(tokens)} to {self.config.max_tokens}")
            tokens = tokens[:self.config.max_tokens]
        return tokens

    def _update_heatmap(self, weights: np.ndarray, tokens: List[str]):
        """Update the heatmap visualization."""
        self.figure.data = []  # Clear previous data
        
        # Create heatmap
        heatmap = go.Heatmap(
            z=weights,
            x=tokens,
            y=tokens,
            colorscale=self.config.colorscale,
            showscale=True,
            name='Attention Weights'
        )
        
        self.figure.add_trace(heatmap)
        
        # Update layout
        self.figure.update_layout(
            xaxis=dict(
                tickangle=45,
                showgrid=False
            ),
            yaxis=dict(
                showgrid=False
            )
        )

    @performance_monitor
    def create_multi_head_view(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: int
    ) -> go.Figure:
        """
        Create multi-head attention visualization.
        
        Args:
            attention_weights: Attention weights [batch, heads, query_len, key_len]
            tokens: Token strings
            layer_idx: Layer index to visualize
            
        Returns:
            Plotly figure with subplots for each attention head
        """
        weights = self._process_attention_weights(attention_weights, layer_idx, None)
        n_heads = weights.shape[0]
        
        # Create subplot grid
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Head {i+1}" for i in range(n_heads)]
        )
        
        # Add heatmaps for each head
        for head_idx in range(n_heads):
            row = head_idx // n_cols + 1
            col = head_idx % n_cols + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=weights[head_idx],
                    x=tokens,
                    y=tokens,
                    colorscale=self.config.colorscale,
                    showscale=head_idx == 0  # Show colorbar only for first head
                ),
                row=row,
                col=col
            )
        
        # Update layout
        fig.update_layout(
            title=f"Multi-head Attention Pattern (Layer {layer_idx})",
            template="plotly_dark",
            height=250 * n_rows,
            showlegend=False
        )
        
        return fig

    def create_layer_comparison(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_indices: List[int]
    ) -> go.Figure:
        """
        Create layer-wise attention pattern comparison.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: Token strings
            layer_indices: List of layer indices to compare
            
        Returns:
            Plotly figure with layer comparison
        """
        n_layers = len(layer_indices)
        
        fig = make_subplots(
            rows=1,
            cols=n_layers,
            subplot_titles=[f"Layer {idx}" for idx in layer_indices]
        )
        
        for i, layer_idx in enumerate(layer_indices):
            weights = self._process_attention_weights(attention_weights, layer_idx, None)
            
            fig.add_trace(
                go.Heatmap(
                    z=weights,
                    x=tokens,
                    y=tokens,
                    colorscale=self.config.colorscale,
                    showscale=i == n_layers-1  # Show colorbar only for last layer
                ),
                row=1,
                col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title="Layer-wise Attention Pattern Comparison",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        return fig
