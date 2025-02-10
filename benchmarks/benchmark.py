import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import coremltools as ct
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
import logging
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    batch_sizes: List[int] = None
    num_iterations: int = 500  # Increased for better statistical significance
    sequence_length: int = 512
    warm_up_iterations: int = 50  # Increased warm-up for stability
    enable_metal_performance_metrics: bool = True
    confidence_thresholds: List[float] = None  # Test different confidence thresholds
    temperatures: List[float] = None  # Test different temperature values
    ensemble_sizes: List[int] = None  # Test different ensemble sizes
    
    def __post_init__(self):
        if self.batch_sizes is None:
            # Optimized batch sizes for accuracy-throughput balance
            self.batch_sizes = [1, 2, 4, 8, 16, 32]
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
        if self.temperatures is None:
            self.temperatures = [0.5, 0.7, 0.9, 1.0]
        if self.ensemble_sizes is None:
            self.ensemble_sizes = [1, 3, 5]

class ModelBenchmark:
    def __init__(
        self,
        pytorch_model_name: str,
        coreml_model_path: str,
        config: BenchmarkConfig = None
    ):
        self.config = config or BenchmarkConfig()
        
        # Load PyTorch model
        self.pytorch_model = AutoModelForSequenceClassification.from_pretrained(pytorch_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pytorch_model_name)
        self.pytorch_model.eval()
        
        # Initialize metrics storage
        self.metrics = {
            'latency': [],
            'throughput': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'confidence_scores': [],
            'batch_sizes': [],
            'temperature_impact': [],
            'ensemble_performance': []
        }
        
        # Initialize performance monitoring
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.pytorch_model.to(self.device)
            self.start_gpu_monitoring()
        else:
            self.device = torch.device('cpu')
            
    def start_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_monitoring = True
        except:
            logger.warning("GPU monitoring not available")
            self.gpu_monitoring = False
            
    def collect_gpu_metrics(self):
        """Collect GPU metrics if monitoring is enabled"""
        if not self.gpu_monitoring:
            return {}
            
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return {
                'memory_used': info.used / info.total,
                'gpu_util': util.gpu
            }
        except:
            return {}
        
        # Load CoreML model
        self.coreml_model = ct.models.MLModel(coreml_model_path)
        
    def run_benchmarks(self, sample_texts: List[str]):
        """Run comprehensive benchmarks with detailed metrics collection"""
        results = []
        
        # Warm-up phase
        logger.info("Starting warm-up phase...")
        for _ in tqdm(range(self.config.warm_up_iterations)):
            _ = self.run_inference(sample_texts[0])
            
        # Main benchmark loop
        logger.info("Starting main benchmark phase...")
        for batch_size in self.config.batch_sizes:
            for temp in self.config.temperatures:
                for conf_threshold in self.config.confidence_thresholds:
                    logger.info(f"Testing batch_size={batch_size}, temp={temp}, threshold={conf_threshold}")
                    
                    batch_metrics = self.benchmark_configuration(
                        sample_texts, 
                        batch_size=batch_size,
                        temperature=temp,
                        confidence_threshold=conf_threshold
                    )
                    results.append(batch_metrics)
                    
        # Save results
        df = pd.DataFrame(results)
        output_path = Path("benchmark_results.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Benchmark results saved to {output_path}")
        
        return df
        
    def benchmark_configuration(self, texts: List[str], batch_size: int, temperature: float, confidence_threshold: float):
        """Run benchmarks for a specific configuration"""
        batch_latencies = []
        batch_throughputs = []
        gpu_metrics = []
        
        # Prepare batches
        num_batches = self.config.num_iterations
        text_batches = [texts[:batch_size] for _ in range(num_batches)]
        
        for batch in tqdm(text_batches, desc=f"Batch size {batch_size}"):
            start_time = time.time()
            
            # Run inference
            outputs = self.run_inference(
                batch,
                temperature=temperature,
                confidence_threshold=confidence_threshold
            )
            
            # Collect timing metrics
            end_time = time.time()
            latency = end_time - start_time
            throughput = batch_size / latency
            
            batch_latencies.append(latency)
            batch_throughputs.append(throughput)
            
            # Collect GPU metrics
            if hasattr(self, 'gpu_monitoring') and self.gpu_monitoring:
                gpu_metrics.append(self.collect_gpu_metrics())
        
        # Aggregate metrics
        metrics = {
            'batch_size': batch_size,
            'temperature': temperature,
            'confidence_threshold': confidence_threshold,
            'avg_latency_ms': np.mean(batch_latencies) * 1000,
            'p50_latency_ms': np.percentile(batch_latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(batch_latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(batch_latencies, 99) * 1000,
            'avg_throughput': np.mean(batch_throughputs),
            'max_throughput': np.max(batch_throughputs)
        }
        
        # Add GPU metrics if available
        if gpu_metrics:
            metrics.update({
                'avg_gpu_util': np.mean([m.get('gpu_util', 0) for m in gpu_metrics]),
                'avg_memory_used': np.mean([m.get('memory_used', 0) for m in gpu_metrics])
            })
            
        return metrics
        
        # Results storage
        self.results = []
        
    def _generate_random_input(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Generate random input data for benchmarking."""
        input_ids = torch.randint(
            0,
            self.tokenizer.vocab_size,
            (batch_size, self.config.sequence_length)
        )
        attention_mask = torch.ones((batch_size, self.config.sequence_length))
        
        return {
            "pytorch": (input_ids, attention_mask),
            "coreml": {
                "input_ids": input_ids.numpy().astype(np.int32),
                "attention_mask": attention_mask.numpy().astype(np.int32)
            }
        }
        
    def _benchmark_pytorch(self, inputs: tuple) -> float:
        """Benchmark PyTorch model inference."""
        start_time = time.perf_counter()
        with torch.no_grad():
            self.pytorch_model(*inputs)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to milliseconds
        
    def _benchmark_coreml(self, inputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Benchmark CoreML model inference with M4-specific metrics."""
        # Warm up the Metal engine
        self.coreml_model.predict(inputs)
        
        # Measure inference time
        start_time = time.perf_counter()
        result = self.coreml_model.predict(inputs)
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        metrics = {
            'inference_time': inference_time,
            'throughput': len(inputs['input_ids']) / (inference_time / 1000),  # samples/second
        }
        
        if self.config.enable_metal_performance_metrics:
            # Get Metal performance metrics if available
            try:
                metal_perf = self.coreml_model.get_metal_performance_statistics()
                metrics.update({
                    'gpu_utilization': metal_perf.get('gpu_utilization', 0),
                    'memory_usage': metal_perf.get('memory_usage', 0),
                    'neural_engine_utilization': metal_perf.get('neural_engine_utilization', 0)
                })
            except Exception as e:
                logger.warning(f"Could not get Metal performance metrics: {e}")
        
        return metrics
        
    def run_benchmarks(self) -> pd.DataFrame:
        """Run comprehensive benchmarks comparing PyTorch and CoreML."""
        logger.info("Starting benchmark run...")
        
        for batch_size in self.config.batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")
            
            pytorch_times = []
            coreml_times = []
            
            # Warm-up runs
            for _ in range(self.config.warm_up_iterations):
                inputs = self._generate_random_input(batch_size)
                self._benchmark_pytorch(inputs["pytorch"])
                self._benchmark_coreml(inputs["coreml"])
            
            # Actual benchmark runs
            for _ in tqdm(range(self.config.num_iterations)):
                inputs = self._generate_random_input(batch_size)
                
                # PyTorch benchmark
                pytorch_time = self._benchmark_pytorch(inputs["pytorch"])
                pytorch_times.append(pytorch_time)
                
                # CoreML benchmark
                coreml_time = self._benchmark_coreml(inputs["coreml"])
                coreml_times.append(coreml_time)
            
            # Calculate statistics
            pytorch_metrics = {
                "batch_size": batch_size,
                "framework": "pytorch",
                "mean_latency": np.mean(pytorch_times),
                "std_latency": np.std(pytorch_times),
                "min_latency": np.min(pytorch_times),
                "max_latency": np.max(pytorch_times),
                "p50_latency": np.percentile(pytorch_times, 50),
                "p95_latency": np.percentile(pytorch_times, 95),
                "p99_latency": np.percentile(pytorch_times, 99),
                "throughput": batch_size / (np.mean(pytorch_times) / 1000)  # samples/second
            }
            self.results.append(pytorch_metrics)
            
            # Process CoreML metrics
            coreml_metrics = {
                "batch_size": batch_size,
                "framework": "coreml",
                "mean_latency": np.mean([m['inference_time'] for m in coreml_times]),
                "std_latency": np.std([m['inference_time'] for m in coreml_times]),
                "min_latency": np.min([m['inference_time'] for m in coreml_times]),
                "max_latency": np.max([m['inference_time'] for m in coreml_times]),
                "p50_latency": np.percentile([m['inference_time'] for m in coreml_times], 50),
                "p95_latency": np.percentile([m['inference_time'] for m in coreml_times], 95),
                "p99_latency": np.percentile([m['inference_time'] for m in coreml_times], 99),
                "mean_throughput": np.mean([m['throughput'] for m in coreml_times])
            }
            
            # Add Metal performance metrics if available
            if self.config.enable_metal_performance_metrics:
                for metric in ['gpu_utilization', 'memory_usage', 'neural_engine_utilization']:
                    values = [m.get(metric, 0) for m in coreml_times]
                    if values:
                        coreml_metrics[f"mean_{metric}"] = np.mean(values)
                        coreml_metrics[f"max_{metric}"] = np.max(values)
            
            self.results.append(coreml_metrics)
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save results
        results_dir = Path("benchmarks/results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        df.to_csv(results_dir / f"benchmark_results_{timestamp}.csv", index=False)
        
        # Generate summary plots
        self._generate_plots(df, results_dir / f"benchmark_plots_{timestamp}")
        
        return df
    
    def _generate_plots(self, df: pd.DataFrame, output_prefix: str) -> None:
        """Generate visualization plots for benchmark results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use("seaborn")
            
            # Latency vs Batch Size
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df,
                x="batch_size",
                y="mean_latency",
                hue="framework",
                marker="o"
            )
            plt.title("Mean Latency vs Batch Size")
            plt.xlabel("Batch Size")
            plt.ylabel("Latency (ms)")
            plt.savefig(f"{output_prefix}_latency.png")
            plt.close()
            
            # Latency Distribution
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=df,
                x="batch_size",
                y="mean_latency",
                hue="framework"
            )
            plt.title("Latency Distribution by Batch Size")
            plt.xlabel("Batch Size")
            plt.ylabel("Latency (ms)")
            plt.savefig(f"{output_prefix}_distribution.png")
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib and seaborn required for plotting. Skipping plot generation.")
