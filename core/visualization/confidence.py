"""
Model confidence visualization and tracking.
Provides real-time visualization of model confidence metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import plotly.graph_objects as go
from collections import deque
from dataclasses import dataclass
from core.metrics import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceConfig:
    """Configuration for confidence visualization."""
    window_size: int = 100  # Number of points to keep in history
    update_interval_ms: int = 500
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    warning_threshold: float = 0.6
    critical_threshold: float = 0.4

class ConfidenceTracker:
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """
        Initialize confidence tracker.
        
        Args:
            config: Optional visualization configuration
        """
        self.config = config or ConfidenceConfig()
        self.figure = None
        self.history = {}  # Dict mapping metrics to their histories
        self.timestamps = deque(maxlen=self.config.window_size)
        self._setup_visualization()
        logger.info("Initialized confidence tracker")

    @performance_monitor
    def _setup_visualization(self):
        """Setup the visualization layout."""
        self.figure = go.Figure()
        self.figure.update_layout(
            title="Model Confidence Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            template="plotly_dark",
            showlegend=True,
            uirevision=True
        )

    @performance_monitor
    def update_confidence(
        self,
        confidence_scores: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> go.Figure:
        """
        Update confidence visualization with new scores.
        
        Args:
            confidence_scores: Dictionary mapping metric names to confidence values
            timestamp: Optional timestamp for the update
            
        Returns:
            Updated Plotly figure
        """
        try:
            # Process and store new scores
            self._update_history(confidence_scores, timestamp)
            
            # Update visualization
            self._update_visualization()
            
            return self.figure
            
        except Exception as e:
            logger.error(f"Error updating confidence: {str(e)}")
            raise

    def _update_history(
        self,
        scores: Dict[str, float],
        timestamp: Optional[float] = None
    ):
        """Update confidence history with new scores."""
        import time
        
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = time.time()
            
        # Update timestamps
        self.timestamps.append(timestamp)
        
        # Update histories for each metric
        for metric, score in scores.items():
            if metric not in self.history:
                self.history[metric] = deque(maxlen=self.config.window_size)
            self.history[metric].append(score)

    def _update_visualization(self):
        """Update the confidence visualization."""
        self.figure.data = []  # Clear previous data
        
        # Add trace for each metric
        for metric, scores in self.history.items():
            self.figure.add_trace(go.Scatter(
                x=list(self.timestamps),
                y=list(scores),
                name=metric,
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        # Add threshold lines
        self.figure.add_hline(
            y=self.config.warning_threshold,
            line_dash="dash",
            line_color="yellow",
            annotation_text="Warning Threshold",
            annotation_position="right"
        )
        
        self.figure.add_hline(
            y=self.config.critical_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold",
            annotation_position="right"
        )
        
        # Update layout
        self.figure.update_layout(
            yaxis=dict(
                range=[
                    self.config.min_confidence - 0.05,
                    self.config.max_confidence + 0.05
                ],
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
        )

    @performance_monitor
    def create_confidence_heatmap(
        self,
        confidence_matrix: np.ndarray,
        x_labels: List[str],
        y_labels: List[str]
    ) -> go.Figure:
        """
        Create confidence heatmap visualization.
        
        Args:
            confidence_matrix: 2D array of confidence scores
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            
        Returns:
            Plotly figure with confidence heatmap
        """
        fig = go.Figure(data=go.Heatmap(
            z=confidence_matrix,
            x=x_labels,
            y=y_labels,
            colorscale='RdYlGn',
            zmin=self.config.min_confidence,
            zmax=self.config.max_confidence
        ))
        
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Categories",
            yaxis_title="Samples",
            template="plotly_dark"
        )
        
        return fig

    @performance_monitor
    def create_confidence_distribution(
        self,
        confidence_scores: Dict[str, List[float]]
    ) -> go.Figure:
        """
        Create confidence distribution visualization.
        
        Args:
            confidence_scores: Dictionary mapping categories to lists of confidence scores
            
        Returns:
            Plotly figure with confidence distributions
        """
        fig = go.Figure()
        
        for category, scores in confidence_scores.items():
            fig.add_trace(go.Violin(
                y=scores,
                name=category,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title="Confidence Score Distribution by Category",
            yaxis_title="Confidence Score",
            template="plotly_dark",
            yaxis=dict(
                range=[
                    self.config.min_confidence - 0.05,
                    self.config.max_confidence + 0.05
                ]
            )
        )
        
        return fig

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for confidence scores."""
        stats = {}
        
        for metric, scores in self.history.items():
            if scores:
                scores_array = np.array(scores)
                stats[metric] = {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array)),
                    'latest': float(scores_array[-1])
                }
        
        return stats
