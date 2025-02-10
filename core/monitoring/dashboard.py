"""Real-time performance monitoring dashboard."""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import threading
import time
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PerformanceDashboard:
    def __init__(self, metrics_dir: str = "metrics", update_interval: int = 1000):
        self.metrics_dir = Path(metrics_dir)
        self.update_interval = update_interval
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("CoreML Model Performance Dashboard"),
            
            # Refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
            
            # Real-time metrics
            html.Div([
                html.H2("Real-time Performance Metrics"),
                html.Div([
                    html.Div([
                        html.H3("Current Latency (ms)"),
                        html.Div(id='current-latency')
                    ], className='metric-box'),
                    html.Div([
                        html.H3("Throughput (req/s)"),
                        html.Div(id='current-throughput')
                    ], className='metric-box'),
                    html.Div([
                        html.H3("Batch Size"),
                        html.Div(id='current-batch-size')
                    ], className='metric-box'),
                ], style={'display': 'flex', 'justifyContent': 'space-around'})
            ]),
            
            # Latency Distribution
            html.Div([
                html.H2("Latency Distribution"),
                dcc.Graph(id='latency-distribution')
            ]),
            
            # Resource Utilization
            html.Div([
                html.H2("Resource Utilization"),
                dcc.Graph(id='resource-utilization')
            ]),
            
            # Batch Processing Metrics
            html.Div([
                html.H2("Batch Processing"),
                dcc.Graph(id='batch-metrics')
            ]),
            
            # Priority Queue Status
            html.Div([
                html.H2("Queue Status"),
                dcc.Graph(id='queue-status')
            ])
        ])
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output('current-latency', 'children'),
             Output('current-throughput', 'children'),
             Output('current-batch-size', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(_):
            try:
                latest_metrics = self._get_latest_metrics()
                if latest_metrics:
                    return [
                        f"{latest_metrics['avg_latency']:.2f}",
                        f"{latest_metrics['throughput']:.2f}",
                        f"{latest_metrics['avg_batch_size']:.1f}"
                    ]
                return ["N/A", "N/A", "N/A"]
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                return ["Error", "Error", "Error"]
        
        @self.app.callback(
            Output('latency-distribution', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_latency_distribution(_):
            try:
                df = self._load_recent_traces()
                if df.empty:
                    return go.Figure()
                    
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df['total_latency'],
                    nbinsx=30,
                    name='Latency Distribution'
                ))
                
                fig.update_layout(
                    title='Request Latency Distribution',
                    xaxis_title='Latency (ms)',
                    yaxis_title='Count'
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating latency distribution: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('resource-utilization', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_resource_utilization(_):
            try:
                df = self._load_recent_batches()
                if df.empty:
                    return go.Figure()
                    
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['gpu_utilization'],
                    name='GPU Utilization'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['memory_utilization'],
                    name='Memory Utilization'
                ))
                
                fig.update_layout(
                    title='Resource Utilization Over Time',
                    xaxis_title='Time',
                    yaxis_title='Utilization (%)'
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating resource utilization: {e}")
                return go.Figure()
                
        @self.app.callback(
            Output('batch-metrics', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_batch_metrics(_):
            try:
                df = self._load_recent_batches()
                if df.empty:
                    return go.Figure()
                    
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['size'],
                    name='Batch Size'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['processing_time'],
                    name='Processing Time'
                ))
                
                fig.update_layout(
                    title='Batch Processing Metrics',
                    xaxis_title='Time',
                    yaxis_title='Value'
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating batch metrics: {e}")
                return go.Figure()
                
        @self.app.callback(
            Output('queue-status', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_queue_status(_):
            try:
                df = self._load_recent_batches()
                if df.empty:
                    return go.Figure()
                    
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['queue_size'],
                    name='Queue Size'
                ))
                
                # Add priority distribution
                priority_counts = df['priorities'].apply(lambda x: pd.Series(x).value_counts()).mean()
                fig.add_trace(go.Bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    name='Priority Distribution'
                ))
                
                fig.update_layout(
                    title='Queue Status and Priority Distribution',
                    xaxis_title='Time / Priority Level',
                    yaxis_title='Count'
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating queue status: {e}")
                return go.Figure()
    
    def _get_latest_metrics(self) -> Dict:
        """Get the latest metrics from the metrics directory."""
        try:
            latest_file = max(self.metrics_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
            with open(latest_file) as f:
                data = json.load(f)
            return data.get('current_stats', {})
        except Exception as e:
            logger.error(f"Error loading latest metrics: {e}")
            return {}
    
    def _load_recent_traces(self) -> pd.DataFrame:
        """Load recent request traces into a DataFrame."""
        try:
            latest_file = max(self.metrics_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
            with open(latest_file) as f:
                data = json.load(f)
            traces = data.get('request_traces', [])
            if not traces:
                return pd.DataFrame()
            return pd.DataFrame(traces)
        except Exception as e:
            logger.error(f"Error loading request traces: {e}")
            return pd.DataFrame()
    
    def _load_recent_batches(self) -> pd.DataFrame:
        """Load recent batch metrics into a DataFrame."""
        try:
            latest_file = max(self.metrics_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
            with open(latest_file) as f:
                data = json.load(f)
            batches = data.get('batch_metrics', [])
            if not batches:
                return pd.DataFrame()
            df = pd.DataFrame(batches)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading batch metrics: {e}")
            return pd.DataFrame()
    
    def run(self, host: str = 'localhost', port: int = 8050, debug: bool = False):
        """Run the dashboard server."""
        self.app.run_server(host=host, port=port, debug=debug)

def start_dashboard(metrics_dir: str = "metrics", port: int = 8050):
    """Start the dashboard in a separate thread."""
    def run_dashboard():
        dashboard = PerformanceDashboard(metrics_dir=metrics_dir)
        dashboard.run(port=port)
    
    thread = threading.Thread(target=run_dashboard, daemon=True)
    thread.start()
    logger.info(f"Dashboard started at http://localhost:{port}")
    return thread
