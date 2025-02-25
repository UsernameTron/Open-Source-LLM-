"""
Token attribution visualization for model interpretability.
Provides real-time visualization of token importance and contributions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import plotly.graph_objects as go
from dataclasses import dataclass
from core.metrics import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class AttributionConfig:
    """Configuration for attribution visualization."""
    method: str = "integrated_gradients"  # or "attention", "lime"
    n_steps: int = 50  # Steps for integrated gradients
    update_interval_ms: int = 200
    colorscale: str = "RdBu"
    baseline_token_id: int = 0  # Usually [PAD] token

class AttributionVisualizer:
    def __init__(self, config: Optional[AttributionConfig] = None):
        """
        Initialize attribution visualizer.
        
        Args:
            config: Optional visualization configuration
        """
        self.config = config or AttributionConfig()
        self.figure = None
        self._setup_visualization()
        logger.info(f"Initialized attribution visualizer with method: {self.config.method}")

    @performance_monitor
    def _setup_visualization(self):
        """Setup the visualization layout."""
        self.figure = go.Figure()
        self.figure.update_layout(
            title="Token Attribution Scores",
            xaxis_title="Tokens",
            yaxis_title="Attribution Score",
            template="plotly_dark",
            showlegend=True,
            uirevision=True
        )

    @performance_monitor
    def visualize_attribution(
        self,
        tokens: List[str],
        scores: Union[torch.Tensor, np.ndarray],
        target_label: Optional[str] = None
    ) -> go.Figure:
        """
        Create token attribution visualization.
        
        Args:
            tokens: List of input tokens
            scores: Attribution scores for each token
            target_label: Optional target label for classification tasks
            
        Returns:
            Plotly figure object
        """
        try:
            # Process attribution scores
            scores = self._process_scores(scores)
            
            # Create visualization
            self._update_visualization(tokens, scores, target_label)
            
            return self.figure
            
        except Exception as e:
            logger.error(f"Error visualizing attribution: {str(e)}")
            raise

    def _process_scores(
        self,
        scores: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """Process attribution scores for visualization."""
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
            
        # Ensure 1D array
        if scores.ndim > 1:
            scores = scores.squeeze()
            
        # Normalize scores
        scores = self._normalize_scores(scores)
        
        return scores

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize attribution scores to [-1, 1] range."""
        abs_max = np.abs(scores).max()
        if abs_max > 0:
            return scores / abs_max
        return scores

    def _update_visualization(
        self,
        tokens: List[str],
        scores: np.ndarray,
        target_label: Optional[str]
    ):
        """Update the attribution visualization."""
        self.figure.data = []  # Clear previous data
        
        # Create bar plot
        bar = go.Bar(
            x=tokens,
            y=scores,
            marker=dict(
                color=scores,
                colorscale=self.config.colorscale,
                showscale=True
            ),
            name='Attribution Score'
        )
        
        self.figure.add_trace(bar)
        
        # Update layout
        title = "Token Attribution Scores"
        if target_label:
            title += f" for {target_label}"
            
        self.figure.update_layout(
            title=title,
            xaxis=dict(
                tickangle=45,
                showgrid=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[-1.1, 1.1]  # Add some padding
            ),
            bargap=0.1
        )

    @performance_monitor
    def create_multi_class_attribution(
        self,
        tokens: List[str],
        scores: Dict[str, np.ndarray]
    ) -> go.Figure:
        """
        Create multi-class attribution visualization.
        
        Args:
            tokens: Input tokens
            scores: Dictionary mapping class labels to attribution scores
            
        Returns:
            Plotly figure with multi-class attribution comparison
        """
        fig = go.Figure()
        
        for label, class_scores in scores.items():
            normalized_scores = self._normalize_scores(class_scores)
            
            fig.add_trace(go.Bar(
                name=label,
                x=tokens,
                y=normalized_scores,
                marker_color=self._get_class_color(label)
            ))
        
        fig.update_layout(
            title="Multi-class Token Attribution Comparison",
            barmode='group',
            xaxis=dict(
                tickangle=45,
                title="Tokens",
                showgrid=False
            ),
            yaxis=dict(
                title="Attribution Score",
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[-1.1, 1.1]
            ),
            template="plotly_dark",
            legend_title="Classes"
        )
        
        return fig

    def _get_class_color(self, label: str) -> str:
        """Get color for class label."""
        # Implementation can be customized based on needs
        colors = {
            "positive": "#2ecc71",
            "negative": "#e74c3c",
            "neutral": "#3498db"
        }
        return colors.get(label.lower(), "#95a5a6")

    @performance_monitor
    def create_temporal_attribution(
        self,
        tokens: List[str],
        scores_over_time: List[np.ndarray],
        timestamps: List[float]
    ) -> go.Figure:
        """
        Create temporal attribution visualization.
        
        Args:
            tokens: Input tokens
            scores_over_time: List of attribution scores at different times
            timestamps: List of corresponding timestamps
            
        Returns:
            Plotly figure showing attribution evolution
        """
        fig = go.Figure()
        
        for i, (scores, time) in enumerate(zip(scores_over_time, timestamps)):
            normalized_scores = self._normalize_scores(scores)
            
            fig.add_trace(go.Bar(
                name=f"t={time:.2f}s",
                x=tokens,
                y=normalized_scores,
                visible=(i == len(scores_over_time) - 1)  # Only show latest by default
            ))
        
        # Add slider
        steps = []
        for i in range(len(scores_over_time)):
            step = dict(
                method="update",
                args=[{"visible": [j == i for j in range(len(scores_over_time))]}],
                label=f"t={timestamps[i]:.2f}s"
            )
            steps.append(step)
        
        sliders = [dict(
            active=len(scores_over_time) - 1,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig.update_layout(
            title="Temporal Attribution Evolution",
            xaxis=dict(
                tickangle=45,
                title="Tokens",
                showgrid=False
            ),
            yaxis=dict(
                title="Attribution Score",
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[-1.1, 1.1]
            ),
            template="plotly_dark",
            sliders=sliders
        )
        
        return fig
