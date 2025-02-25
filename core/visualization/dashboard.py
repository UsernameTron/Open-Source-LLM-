"""
Interactive dashboard for real-time LLM visualization.
Integrates attention, attribution, and confidence visualizations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from core.metrics import performance_monitor
from .attention import AttentionVisualizer, AttentionConfig
from .attribution import AttributionVisualizer, AttributionConfig
from .confidence import ConfidenceTracker, ConfidenceConfig

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for visualization dashboard."""
    attention_config: Optional[AttentionConfig] = None
    attribution_config: Optional[AttributionConfig] = None
    confidence_config: Optional[ConfidenceConfig] = None
    update_interval_ms: int = 1000
    layout: str = "grid"  # or "vertical", "horizontal"

class Dashboard:
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize visualization dashboard.
        
        Args:
            config: Optional dashboard configuration
        """
        self.config = config or DashboardConfig()
        
        # Initialize visualizers
        self.attention_viz = AttentionVisualizer(self.config.attention_config)
        self.attribution_viz = AttributionVisualizer(self.config.attribution_config)
        self.confidence_viz = ConfidenceTracker(self.config.confidence_config)
        
        self.figure = None
        self._setup_dashboard()
        logger.info("Initialized visualization dashboard")

    @performance_monitor
    def _setup_dashboard(self):
        """Setup the dashboard layout."""
        if self.config.layout == "grid":
            self.figure = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Attention Patterns",
                    "Token Attribution",
                    "Confidence Over Time",
                    "Model Statistics"
                )
            )
        else:
            self.figure = make_subplots(
                rows=3 if self.config.layout == "vertical" else 1,
                cols=1 if self.config.layout == "vertical" else 3
            )
        
        self.figure.update_layout(
            title="LLM Visualization Dashboard",
            template="plotly_dark",
            showlegend=True,
            height=1000 if self.config.layout == "vertical" else 400,
            width=1200
        )

    @performance_monitor
    def update_dashboard(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        tokens: Optional[List[str]] = None,
        attribution_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None
    ) -> go.Figure:
        """
        Update all dashboard components.
        
        Args:
            attention_weights: Optional attention weight tensor
            tokens: Optional list of tokens
            attribution_scores: Optional attribution scores
            confidence_scores: Optional confidence scores
            timestamp: Optional timestamp for the update
            
        Returns:
            Updated Plotly figure
        """
        try:
            # Clear previous data
            self.figure.data = []
            
            # Update attention visualization
            if attention_weights is not None and tokens is not None:
                self._update_attention(attention_weights, tokens)
            
            # Update attribution visualization
            if attribution_scores is not None and tokens is not None:
                self._update_attribution(attribution_scores, tokens)
            
            # Update confidence visualization
            if confidence_scores is not None:
                self._update_confidence(confidence_scores, timestamp)
            
            # Update statistics
            self._update_statistics()
            
            return self.figure
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
            raise

    def _update_attention(
        self,
        weights: torch.Tensor,
        tokens: List[str]
    ):
        """Update attention visualization in dashboard."""
        attention_fig = self.attention_viz.visualize_attention(weights, tokens)
        
        for trace in attention_fig.data:
            if self.config.layout == "grid":
                self.figure.add_trace(trace, row=1, col=1)
            else:
                row = 1 if self.config.layout == "horizontal" else 1
                col = 1 if self.config.layout == "vertical" else 1
                self.figure.add_trace(trace, row=row, col=col)

    def _update_attribution(
        self,
        scores: Union[torch.Tensor, np.ndarray],
        tokens: List[str]
    ):
        """Update attribution visualization in dashboard."""
        attribution_fig = self.attribution_viz.visualize_attribution(tokens, scores)
        
        for trace in attribution_fig.data:
            if self.config.layout == "grid":
                self.figure.add_trace(trace, row=1, col=2)
            else:
                row = 1 if self.config.layout == "horizontal" else 2
                col = 2 if self.config.layout == "vertical" else 1
                self.figure.add_trace(trace, row=row, col=col)

    def _update_confidence(
        self,
        scores: Dict[str, float],
        timestamp: Optional[float]
    ):
        """Update confidence visualization in dashboard."""
        confidence_fig = self.confidence_viz.update_confidence(scores, timestamp)
        
        for trace in confidence_fig.data:
            if self.config.layout == "grid":
                self.figure.add_trace(trace, row=2, col=1)
            else:
                row = 1 if self.config.layout == "horizontal" else 3
                col = 3 if self.config.layout == "vertical" else 1
                self.figure.add_trace(trace, row=row, col=col)

    def _update_statistics(self):
        """Update model statistics visualization."""
        stats = self.confidence_viz.get_summary_statistics()
        
        # Create table
        table_data = []
        headers = ['Metric', 'Latest', 'Mean', 'Std', 'Min', 'Max']
        table_data.append(headers)
        
        for metric, metric_stats in stats.items():
            row = [
                metric,
                f"{metric_stats['latest']:.3f}",
                f"{metric_stats['mean']:.3f}",
                f"{metric_stats['std']:.3f}",
                f"{metric_stats['min']:.3f}",
                f"{metric_stats['max']:.3f}"
            ]
            table_data.append(row)
        
        table = go.Table(
            header=dict(
                values=headers,
                fill_color='rgba(64, 64, 64, 0.8)',
                align='left'
            ),
            cells=dict(
                values=list(zip(*table_data[1:])),
                fill_color='rgba(32, 32, 32, 0.8)',
                align='left'
            )
        )
        
        if self.config.layout == "grid":
            self.figure.add_trace(table, row=2, col=2)
        else:
            row = 1 if self.config.layout == "horizontal" else 3
            col = 3 if self.config.layout == "vertical" else 1
            self.figure.add_trace(table, row=row, col=col)

    @performance_monitor
    def create_summary_view(self) -> go.Figure:
        """
        Create summary visualization of all components.
        
        Returns:
            Plotly figure with summary view
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Attention Overview",
                "Attribution Summary",
                "Confidence Trends",
                "Performance Metrics"
            ]
        )
        
        # Add summary visualizations
        stats = self.confidence_viz.get_summary_statistics()
        
        # Create summary traces
        for metric, metric_stats in stats.items():
            # Confidence trend
            fig.add_trace(
                go.Scatter(
                    y=list(self.confidence_viz.history[metric]),
                    name=f"{metric} Trend",
                    mode='lines'
                ),
                row=2,
                col=1
            )
            
            # Performance metrics
            fig.add_trace(
                go.Bar(
                    x=[metric],
                    y=[metric_stats['latest']],
                    name=f"{metric} Latest"
                ),
                row=2,
                col=2
            )
        
        fig.update_layout(
            title="Model Performance Summary",
            template="plotly_dark",
            showlegend=True,
            height=800
        )
        
        return fig
