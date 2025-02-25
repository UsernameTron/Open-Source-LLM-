"""
Example benchmark comparing vector store performance across Apple Silicon variants.
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from core.thermal.monitor import ThermalMonitor
from core.thermal.profile import ProfileManager
from core.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from core.benchmark.report import ReportGenerator, PerformanceVisualizer
from core.storage.vector.store import VectorStore
from core.storage.vector.index import VectorIndex
from core.storage.types import VectorConfig, SearchConfig, VectorMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def benchmark_vector_operations(
    vector_store: VectorStore,
    num_vectors: int = 10000,
    vector_dim: int = 768,
    batch_size: int = 32
) -> dict:
    """Run vector store benchmark operations."""
    
    # Generate test data
    vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)
    metadata_list = [
        VectorMetadata(
            id=str(i),
            text=f"Text {i}",
            metadata={"index": i}
        )
        for i in range(num_vectors)
    ]
    
    # Store vectors
    await vector_store.store(vectors, metadata_list)
    
    # Generate query vector
    query = np.random.randn(vector_dim).astype(np.float32)
    
    # Search vectors
    results = await vector_store.search(
        query,
        "test query",
        SearchConfig(top_k=10)
    )
    
    return {
        'batch_size': batch_size,
        'num_results': len(results)
    }

async def main():
    # Initialize components
    thermal_monitor = ThermalMonitor()
    profile_manager = ProfileManager(thermal_monitor)
    benchmark_runner = BenchmarkRunner(thermal_monitor, profile_manager)
    report_generator = ReportGenerator()
    visualizer = PerformanceVisualizer()
    
    # Create output directory
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define benchmark configurations
    configs = [
        BenchmarkConfig(
            name="balanced_profile",
            description="Vector operations with balanced profile",
            iterations=5,
            warmup_iterations=1,
            thermal_profile="balanced"
        ),
        BenchmarkConfig(
            name="performance_profile",
            description="Vector operations with performance profile",
            iterations=5,
            warmup_iterations=1,
            thermal_profile="performance"
        ),
        BenchmarkConfig(
            name="efficiency_profile",
            description="Vector operations with efficiency profile",
            iterations=5,
            warmup_iterations=1,
            thermal_profile="efficiency"
        )
    ]
    
    # Run benchmarks
    results = []
    for config in configs:
        logger.info(f"Running benchmark: {config.name}")
        
        # Create vector store for this run
        vector_store = VectorStore(
            VectorConfig(
                dimension=768,
                metric="cosine",
                index_type="hnsw"
            )
        )
        
        # Run benchmark
        result = await benchmark_runner.run_benchmark(
            config,
            benchmark_vector_operations,
            vector_store
        )
        results.append(result)
        
        # Cleanup
        await vector_store.close()
        
    # Generate reports
    reports = report_generator.generate_report(
        results,
        baseline_name="balanced_profile"
    )
    
    # Save reports
    report_path = output_dir / f"benchmark_report_{timestamp}.json"
    report_generator.save_report(reports, str(report_path))
    
    # Generate comparison table
    table = report_generator.generate_comparison_table(
        reports,
        baseline_name="balanced_profile"
    )
    table_path = output_dir / f"comparison_table_{timestamp}.txt"
    with open(table_path, 'w') as f:
        f.write(table)
        
    # Create visualization dashboard
    dashboard_path = output_dir / f"dashboard_{timestamp}.html"
    visualizer.create_performance_dashboard(reports, str(dashboard_path))
    
    logger.info(f"Benchmark results saved to {output_dir}")
    
if __name__ == "__main__":
    asyncio.run(main())
