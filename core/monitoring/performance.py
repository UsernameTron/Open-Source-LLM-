"""Performance monitoring and visualization for the inference engine."""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class PerformanceReport:
    timestamp: str
    batch_sizes: List[int]
    latencies: List[float]
    throughputs: List[float]
    gpu_utilization: List[float]
    memory_utilization: List[float]
    queue_sizes: List[int]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "metrics": {
                "batch_sizes": self.batch_sizes,
                "latencies": self.latencies,
                "throughputs": self.throughputs,
                "gpu_utilization": self.gpu_utilization,
                "memory_utilization": self.memory_utilization,
                "queue_sizes": self.queue_sizes
            }
        }
        
class PerformanceMonitor:
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_window = []
        
    def update_metrics(self, metrics: Dict):
        """Update the metrics window with new metrics."""
        self.metrics_window.append({
            **metrics,
            'timestamp': time.time()
        })
        
    def generate_report(self) -> PerformanceReport:
        """Generate a performance report from metrics window."""
        if not self.metrics_window:
            return None
            
        # Extract metrics with fallbacks
        batch_sizes = [m.get('batch_size', 1) for m in self.metrics_window]
        latencies = []
        for m in self.metrics_window:
            # Try different possible latency keys
            if 'latency_ms' in m:
                latencies.append(m['latency_ms'])
            elif 'latency' in m:
                latencies.append(m['latency'])
            else:
                # Default to time difference if no latency found
                latencies.append(0.0)
        
        # Calculate derived metrics
        throughputs = []
        for i in range(len(self.metrics_window)):
            window = self.metrics_window[max(0, i-10):i+1]
            total_requests = sum(m.get('batch_size', 1) for m in window)
            time_span = window[-1].get('timestamp', time.time()) - window[0].get('timestamp', time.time()-0.001)
            throughputs.append(total_requests / max(time_span, 0.001))
            
        return PerformanceReport(
            timestamp=datetime.now().isoformat(),
            batch_sizes=batch_sizes,
            latencies=latencies,
            throughputs=throughputs,
            gpu_utilization=[m.get('gpu_utilization', 0) for m in self.metrics_window],
            memory_utilization=[m.get('memory_utilization', 0) for m in self.metrics_window],
            queue_sizes=[m.get('queue_size', 0) for m in self.metrics_window]
        )
        
    def plot_metrics(self, save_path: Optional[str] = None):
        """Generate performance visualization plots."""
        if not self.metrics_window:
            logger.warning("No metrics to plot")
            return
            
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics_window)
        
        # Ensure numeric columns
        numeric_cols = ['latency_ms', 'batch_size', 'gpu_utilization', 'memory_utilization', 'timestamp']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime for plotting
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Latency over time
        if 'latency_ms' in df.columns:
            sns.lineplot(data=df, x='datetime', y='latency_ms', ax=ax1)
            ax1.set_title('Latency Over Time')
            ax1.set_ylabel('Latency (ms)')
        
        # Plot 2: Throughput over time
        if 'batch_size' in df.columns:
            window_size = 10
            df['throughput'] = df['batch_size'].rolling(window=window_size).sum() / \
                              (df['timestamp'].diff().rolling(window=window_size).sum().fillna(0.001))
            sns.lineplot(data=df, x='datetime', y='throughput', ax=ax2)
            ax2.set_title('Throughput Over Time')
            ax2.set_ylabel('Requests/second')
        
        # Plot 3: Resource utilization
        if 'gpu_utilization' in df.columns:
            sns.lineplot(data=df, x='datetime', y='gpu_utilization', ax=ax3, label='GPU')
        if 'memory_utilization' in df.columns:
            sns.lineplot(data=df, x='datetime', y='memory_utilization', ax=ax3, label='Memory')
        ax3.set_title('Resource Utilization Over Time')
        ax3.set_ylabel('Utilization %')
        ax3.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(self.output_dir / f'performance_plot_{int(time.time())}.png')
        plt.close()

    def save_report(self, report: PerformanceReport):
        """Save performance report to file."""
        if not report:
            logger.warning("No report to save")
            return
            
        # Create report filename with timestamp
        filename = f'performance_report_{int(time.time())}.json'
        filepath = self.output_dir / filename
        
        # Save report as JSON
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
            
        logger.info(f"Saved performance report to {filepath}")
