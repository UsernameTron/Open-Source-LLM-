"""
Benchmark report generation and analysis.
"""

import logging
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from scipy import stats
from tabulate import tabulate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.metrics import performance_monitor
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""
    # Basic information
    name: str
    description: str
    timestamp: str
    duration_seconds: float
    
    # Performance summary
    operations_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Resource utilization
    avg_cpu_util: float
    avg_gpu_util: float
    avg_memory_gb: float
    avg_power_w: float
    
    # Efficiency metrics
    compute_efficiency: float
    memory_efficiency: float
    energy_efficiency: float
    
    # Statistical analysis
    latency_std_dev: float
    throughput_std_dev: float
    confidence_interval: tuple[float, float]
    
    # Raw data for detailed analysis
    raw_data: Dict

class ReportGenerator:
    """Generates benchmark reports and visualizations."""
    
    def __init__(self):
        """Initialize report generator."""
        logger.info("Initialized report generator")
        
    @performance_monitor
    def generate_report(
        self,
        results: List[BenchmarkResult],
        baseline_name: Optional[str] = None
    ) -> List[BenchmarkReport]:
        """
        Generate reports from benchmark results.
        
        Args:
            results: List of benchmark results
            baseline_name: Optional name of baseline for comparison
            
        Returns:
            List of benchmark reports
        """
        try:
            reports = []
            baseline = None
            
            # Find baseline if specified
            if baseline_name:
                for result in results:
                    if result.config.name == baseline_name:
                        baseline = result
                        break
                        
            # Generate report for each result
            for result in results:
                # Create dataframe from raw data
                df = pd.DataFrame(result.raw_data['operation_times'])
                
                # Calculate statistics
                latencies = df['operation_times'] * 1000  # Convert to ms
                throughput = 1 / df['operation_times']
                
                report = BenchmarkReport(
                    name=result.config.name,
                    description=result.config.description,
                    timestamp=result.start_time,
                    duration_seconds=result.duration_seconds,
                    
                    # Performance metrics
                    operations_per_second=result.performance.operations_per_second,
                    avg_latency_ms=latencies.mean(),
                    p95_latency_ms=latencies.quantile(0.95),
                    p99_latency_ms=latencies.quantile(0.99),
                    
                    # Resource utilization
                    avg_cpu_util=result.resources.cpu_util_percent if result.resources else 0,
                    avg_gpu_util=result.resources.gpu_util_percent if result.resources else 0,
                    avg_memory_gb=result.resources.memory_used_gb if result.resources else 0,
                    avg_power_w=result.resources.total_power_w if result.resources else 0,
                    
                    # Efficiency metrics
                    compute_efficiency=result.performance.compute_efficiency,
                    memory_efficiency=result.performance.memory_efficiency,
                    energy_efficiency=result.performance.energy_efficiency,
                    
                    # Statistical analysis
                    latency_std_dev=latencies.std(),
                    throughput_std_dev=throughput.std(),
                    confidence_interval=stats.t.interval(
                        0.95,
                        len(latencies) - 1,
                        loc=latencies.mean(),
                        scale=stats.sem(latencies)
                    ),
                    
                    # Raw data
                    raw_data=result.raw_data
                )
                
                reports.append(report)
                
            return reports
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
            
    @performance_monitor
    def save_report(
        self,
        reports: List[BenchmarkReport],
        output_path: str
    ):
        """
        Save reports to file.
        
        Args:
            reports: List of benchmark reports
            output_path: Path to save report
        """
        try:
            # Convert reports to dict
            report_data = [asdict(report) for report in reports]
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            logger.info(f"Saved report to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise
            
    @performance_monitor
    def generate_comparison_table(
        self,
        reports: List[BenchmarkReport],
        baseline_name: Optional[str] = None
    ) -> str:
        """
        Generate comparison table.
        
        Args:
            reports: List of benchmark reports
            baseline_name: Optional baseline name
            
        Returns:
            Formatted table string
        """
        try:
            # Prepare data
            data = []
            baseline = None
            
            if baseline_name:
                for report in reports:
                    if report.name == baseline_name:
                        baseline = report
                        break
                        
            # Create rows
            for report in reports:
                row = [
                    report.name,
                    f"{report.operations_per_second:.2f}",
                    f"{report.avg_latency_ms:.2f}",
                    f"{report.p95_latency_ms:.2f}",
                    f"{report.p99_latency_ms:.2f}",
                    f"{report.avg_cpu_util:.1f}%",
                    f"{report.avg_gpu_util:.1f}%",
                    f"{report.avg_memory_gb:.1f}",
                    f"{report.avg_power_w:.1f}",
                    f"{report.compute_efficiency:.2f}",
                    f"{report.energy_efficiency:.2f}"
                ]
                
                # Add comparison if baseline exists
                if baseline and report != baseline:
                    speedup = (
                        report.operations_per_second /
                        baseline.operations_per_second
                    )
                    row.append(f"{speedup:.2f}x")
                else:
                    row.append("-")
                    
                data.append(row)
                
            # Create table
            headers = [
                "Model",
                "Ops/sec",
                "Avg Latency",
                "P95 Latency",
                "P99 Latency",
                "CPU Util",
                "GPU Util",
                "Memory (GB)",
                "Power (W)",
                "Compute Eff",
                "Energy Eff",
                "vs Baseline"
            ]
            
            return tabulate(
                data,
                headers=headers,
                tablefmt="grid"
            )
            
        except Exception as e:
            logger.error(f"Error generating comparison table: {str(e)}")
            raise
            
class PerformanceVisualizer:
    """Visualizes benchmark performance data."""
    
    def __init__(self):
        """Initialize visualizer."""
        logger.info("Initialized performance visualizer")
        
    @performance_monitor
    def create_performance_dashboard(
        self,
        reports: List[BenchmarkReport],
        output_path: str
    ):
        """
        Create interactive performance dashboard.
        
        Args:
            reports: List of benchmark reports
            output_path: Path to save dashboard HTML
        """
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    "Throughput Comparison",
                    "Latency Distribution",
                    "Resource Utilization",
                    "Power Consumption",
                    "Efficiency Metrics",
                    "Performance Timeline"
                )
            )
            
            # Add throughput comparison
            self._add_throughput_plot(fig, reports, row=1, col=1)
            
            # Add latency distribution
            self._add_latency_plot(fig, reports, row=1, col=2)
            
            # Add resource utilization
            self._add_resource_plot(fig, reports, row=2, col=1)
            
            # Add power consumption
            self._add_power_plot(fig, reports, row=2, col=2)
            
            # Add efficiency metrics
            self._add_efficiency_plot(fig, reports, row=3, col=1)
            
            # Add performance timeline
            self._add_timeline_plot(fig, reports, row=3, col=2)
            
            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                showlegend=True,
                title_text="Performance Benchmark Results"
            )
            
            # Save dashboard
            fig.write_html(output_path)
            logger.info(f"Saved dashboard to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise
            
    def _add_throughput_plot(
        self,
        fig: go.Figure,
        reports: List[BenchmarkReport],
        row: int,
        col: int
    ):
        """Add throughput comparison plot."""
        names = [r.name for r in reports]
        throughput = [r.operations_per_second for r in reports]
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=throughput,
                name="Throughput"
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Model", row=row, col=col)
        fig.update_yaxes(title_text="Operations/second", row=row, col=col)
        
    def _add_latency_plot(
        self,
        fig: go.Figure,
        reports: List[BenchmarkReport],
        row: int,
        col: int
    ):
        """Add latency distribution plot."""
        for report in reports:
            latencies = [
                t * 1000 for t in report.raw_data['operation_times']
            ]
            
            fig.add_trace(
                go.Box(
                    y=latencies,
                    name=report.name,
                    boxpoints='outliers'
                ),
                row=row,
                col=col
            )
            
        fig.update_xaxes(title_text="Model", row=row, col=col)
        fig.update_yaxes(title_text="Latency (ms)", row=row, col=col)
        
    def _add_resource_plot(
        self,
        fig: go.Figure,
        reports: List[BenchmarkReport],
        row: int,
        col: int
    ):
        """Add resource utilization plot."""
        names = [r.name for r in reports]
        cpu_util = [r.avg_cpu_util for r in reports]
        gpu_util = [r.avg_gpu_util for r in reports]
        memory_util = [r.avg_memory_gb for r in reports]
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=cpu_util,
                name="CPU Utilization"
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=gpu_util,
                name="GPU Utilization"
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=memory_util,
                name="Memory Usage (GB)"
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Model", row=row, col=col)
        fig.update_yaxes(title_text="Utilization (%)", row=row, col=col)
        
    def _add_power_plot(
        self,
        fig: go.Figure,
        reports: List[BenchmarkReport],
        row: int,
        col: int
    ):
        """Add power consumption plot."""
        for report in reports:
            if 'resource_samples' in report.raw_data:
                times = list(range(len(report.raw_data['resource_samples'])))
                power = [
                    sample['total_power_w']
                    for sample in report.raw_data['resource_samples']
                ]
                
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=power,
                        name=f"{report.name} Power"
                    ),
                    row=row,
                    col=col
                )
                
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Power (W)", row=row, col=col)
        
    def _add_efficiency_plot(
        self,
        fig: go.Figure,
        reports: List[BenchmarkReport],
        row: int,
        col: int
    ):
        """Add efficiency metrics plot."""
        names = [r.name for r in reports]
        compute_eff = [r.compute_efficiency for r in reports]
        memory_eff = [r.memory_efficiency for r in reports]
        energy_eff = [r.energy_efficiency for r in reports]
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=compute_eff,
                name="Compute Efficiency"
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=memory_eff,
                name="Memory Efficiency"
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=energy_eff,
                name="Energy Efficiency"
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Model", row=row, col=col)
        fig.update_yaxes(title_text="Efficiency Score", row=row, col=col)
        
    def _add_timeline_plot(
        self,
        fig: go.Figure,
        reports: List[BenchmarkReport],
        row: int,
        col: int
    ):
        """Add performance timeline plot."""
        for report in reports:
            times = list(range(len(report.raw_data['operation_times'])))
            latencies = [
                t * 1000 for t in report.raw_data['operation_times']
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=latencies,
                    name=f"{report.name} Latency"
                ),
                row=row,
                col=col
            )
            
        fig.update_xaxes(title_text="Operation #", row=row, col=col)
        fig.update_yaxes(title_text="Latency (ms)", row=row, col=col)
