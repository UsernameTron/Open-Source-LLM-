"""Unified metrics collection system."""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point."""
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collect and aggregate metrics from all subsystems."""
    
    def __init__(
        self,
        history_size: int = 1000,
        flush_interval: float = 60.0
    ):
        self.history_size = history_size
        self.flush_interval = flush_interval
        self.metrics: Dict[str, Dict[str, deque]] = {
            'document_processing': {},
            'streaming': {},
            'inference': {},
            'system': {}
        }
        self._flush_task = None
        
    async def start(self):
        """Start metrics collection."""
        if self._flush_task:
            return
            
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Started metrics collection")
        
    async def stop(self):
        """Stop metrics collection."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
            logger.info("Stopped metrics collection")
            
    def update_metric(
        self,
        category: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update a specific metric."""
        try:
            if category not in self.metrics:
                self.metrics[category] = {}
                
            if metric_name not in self.metrics[category]:
                self.metrics[category][metric_name] = deque(
                    maxlen=self.history_size
                )
                
            point = MetricPoint(
                value=value,
                metadata=metadata or {}
            )
            self.metrics[category][metric_name].append(point)
            
        except Exception as e:
            logger.error(f"Error updating metric: {str(e)}")
            
    def get_metrics(
        self,
        category: Optional[str] = None,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get metrics with optional filtering."""
        try:
            result = {}
            
            categories = [category] if category else self.metrics.keys()
            
            for cat in categories:
                if cat not in self.metrics:
                    continue
                    
                result[cat] = {}
                metrics = self.metrics[cat]
                
                for name, points in metrics.items():
                    if metric_names and name not in metric_names:
                        continue
                        
                    filtered_points = []
                    for point in points:
                        if start_time and point.timestamp < start_time:
                            continue
                        if end_time and point.timestamp > end_time:
                            continue
                        filtered_points.append({
                            'value': point.value,
                            'timestamp': point.timestamp,
                            'metadata': point.metadata
                        })
                        
                    if filtered_points:
                        result[cat][name] = filtered_points
                        
            return result
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {}
            
    def get_latest(
        self,
        category: str,
        metric_name: str
    ) -> Optional[MetricPoint]:
        """Get latest value for a specific metric."""
        try:
            if (category in self.metrics and
                metric_name in self.metrics[category] and
                self.metrics[category][metric_name]):
                return self.metrics[category][metric_name][-1]
            return None
        except Exception as e:
            logger.error(f"Error getting latest metric: {str(e)}")
            return None
            
    def get_statistics(
        self,
        category: str,
        metric_name: str
    ) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        try:
            if (category not in self.metrics or
                metric_name not in self.metrics[category]):
                return {}
                
            points = self.metrics[category][metric_name]
            if not points:
                return {}
                
            values = [p.value for p in points]
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'count': len(values)
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}
            
    async def _flush_loop(self):
        """Periodically flush old metrics."""
        while True:
            try:
                current_time = time.time()
                for category in self.metrics.values():
                    for metric_queue in category.values():
                        while (metric_queue and
                               current_time - metric_queue[0].timestamp >
                               self.flush_interval):
                            metric_queue.popleft()
                            
                await asyncio.sleep(self.flush_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics flush loop: {str(e)}")
                await asyncio.sleep(self.flush_interval * 2)
                
    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()
