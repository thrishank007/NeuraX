"""
Performance benchmarking and optimization for SecureInsight
"""
import time
import statistics
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from loguru import logger
from contextlib import contextmanager

from .memory_manager import MemoryManager, MemoryStats


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    operation_name: str
    execution_time: float
    memory_before: MemoryStats
    memory_after: MemoryStats
    memory_peak: MemoryStats
    cpu_usage: List[float]
    throughput: Optional[float] = None
    items_processed: Optional[int] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LoadTestResult:
    """Results from a load test"""
    test_name: str
    total_items: int
    successful_items: int
    failed_items: int
    total_time: float
    average_time_per_item: float
    throughput_items_per_second: float
    memory_usage_peak: float
    cpu_usage_average: float
    error_rate: float
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)


class PerformanceBenchmarker:
    """Performance benchmarking and monitoring"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.benchmark_history = []
        self.monitoring_active = False
        self.current_benchmark_stats = {}
        
    @contextmanager
    def benchmark_operation(self, operation_name: str, items_count: Optional[int] = None):
        """
        Context manager for benchmarking operations
        
        Args:
            operation_name: Name of the operation being benchmarked
            items_count: Number of items being processed (for throughput calculation)
        """
        # Start monitoring
        memory_before = self.memory_manager.get_memory_stats()
        cpu_usage = []
        memory_peak = memory_before
        
        # Start CPU monitoring thread
        monitoring_active = True
        
        def monitor_resources():
            nonlocal memory_peak, cpu_usage, monitoring_active
            while monitoring_active:
                try:
                    cpu_usage.append(psutil.cpu_percent(interval=0.1))
                    current_memory = self.memory_manager.get_memory_stats()
                    if current_memory.memory_percent > memory_peak.memory_percent:
                        memory_peak = current_memory
                except Exception as e:
                    logger.error(f"Error monitoring resources: {e}")
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        # Record start time
        start_time = time.time()
        error_count = 0
        
        try:
            yield self
        except Exception as e:
            error_count += 1
            logger.error(f"Error in benchmarked operation {operation_name}: {e}")
            raise
        finally:
            # Stop monitoring
            monitoring_active = False
            monitor_thread.join(timeout=1)
            
            # Record end time and memory
            end_time = time.time()
            execution_time = end_time - start_time
            memory_after = self.memory_manager.get_memory_stats()
            
            # Calculate throughput
            throughput = None
            if items_count and execution_time > 0:
                throughput = items_count / execution_time
            
            # Create benchmark result
            result = BenchmarkResult(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                cpu_usage=cpu_usage,
                throughput=throughput,
                items_processed=items_count,
                error_count=error_count
            )
            
            # Store result
            self.benchmark_history.append(result)
            
            # Log summary
            logger.info(f"Benchmark '{operation_name}': {execution_time:.2f}s, "
                       f"Memory: {memory_before.memory_percent:.1f}% -> {memory_after.memory_percent:.1f}% "
                       f"(peak: {memory_peak.memory_percent:.1f}%)")
            
            if throughput:
                logger.info(f"Throughput: {throughput:.2f} items/second")
    
    def benchmark_embedding_generation(self, embedding_func: Callable, 
                                     test_data: List[Any], 
                                     batch_sizes: List[int] = None) -> Dict[int, BenchmarkResult]:
        """
        Benchmark embedding generation with different batch sizes
        
        Args:
            embedding_func: Function to generate embeddings
            test_data: Test data for embedding generation
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary mapping batch size to benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64]
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking embedding generation with batch size: {batch_size}")
            
            with self.benchmark_operation(f"embedding_generation_batch_{batch_size}", 
                                        len(test_data)) as benchmark:
                try:
                    # Process in batches
                    for i in range(0, len(test_data), batch_size):
                        batch = test_data[i:i + batch_size]
                        embedding_func(batch)
                    
                    results[batch_size] = self.benchmark_history[-1]
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark batch size {batch_size}: {e}")
        
        return results
    
    def benchmark_vector_search(self, vector_store, query_embeddings: List[np.ndarray],
                               k_values: List[int] = None) -> Dict[int, BenchmarkResult]:
        """
        Benchmark vector search with different k values
        
        Args:
            vector_store: Vector store instance
            query_embeddings: List of query embeddings
            k_values: List of k values to test
            
        Returns:
            Dictionary mapping k value to benchmark results
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50]
        
        results = {}
        
        for k in k_values:
            logger.info(f"Benchmarking vector search with k={k}")
            
            with self.benchmark_operation(f"vector_search_k_{k}", 
                                        len(query_embeddings)) as benchmark:
                try:
                    for query_embedding in query_embeddings:
                        vector_store.similarity_search(query_embedding, k=k)
                    
                    results[k] = self.benchmark_history[-1]
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark k={k}: {e}")
        
        return results
    
    def benchmark_memory_operations(self, data_sizes: List[int]) -> Dict[int, BenchmarkResult]:
        """
        Benchmark memory operations with different data sizes
        
        Args:
            data_sizes: List of data sizes to test (in MB)
            
        Returns:
            Dictionary mapping data size to benchmark results
        """
        results = {}
        
        for size_mb in data_sizes:
            logger.info(f"Benchmarking memory operations with {size_mb}MB data")
            
            # Calculate array size for the target memory usage
            elements = int((size_mb * 1024 * 1024) / 4)  # 4 bytes per float32
            
            with self.benchmark_operation(f"memory_ops_{size_mb}MB", elements) as benchmark:
                try:
                    # Create large array
                    data = np.random.random((elements,)).astype(np.float32)
                    
                    # Perform some operations
                    normalized = data / np.linalg.norm(data)
                    similarity = np.dot(normalized, normalized.T)
                    
                    # Clean up
                    del data, normalized, similarity
                    
                    results[size_mb] = self.benchmark_history[-1]
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark {size_mb}MB: {e}")
        
        return results
    
    def run_load_test(self, test_name: str, test_function: Callable,
                     test_data: List[Any], batch_size: int = 10,
                     max_errors: int = 10) -> LoadTestResult:
        """
        Run a load test with large dataset
        
        Args:
            test_name: Name of the load test
            test_function: Function to test
            test_data: Test data
            batch_size: Batch size for processing
            max_errors: Maximum errors before stopping test
            
        Returns:
            Load test results
        """
        logger.info(f"Starting load test: {test_name} with {len(test_data)} items")
        
        start_time = time.time()
        successful_items = 0
        failed_items = 0
        benchmark_results = []
        cpu_usage_samples = []
        memory_peak = 0
        
        # Start resource monitoring
        monitoring_active = True
        
        def monitor_resources():
            nonlocal memory_peak, cpu_usage_samples, monitoring_active
            while monitoring_active:
                try:
                    cpu_usage_samples.append(psutil.cpu_percent(interval=0.5))
                    memory_stats = self.memory_manager.get_memory_stats()
                    memory_peak = max(memory_peak, memory_stats.memory_percent)
                except Exception:
                    pass
                time.sleep(0.5)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        try:
            # Process data in batches
            for i in range(0, len(test_data), batch_size):
                if failed_items >= max_errors:
                    logger.warning(f"Stopping load test due to {failed_items} errors")
                    break
                
                batch = test_data[i:i + batch_size]
                batch_name = f"{test_name}_batch_{i//batch_size + 1}"
                
                with self.benchmark_operation(batch_name, len(batch)) as benchmark:
                    try:
                        test_function(batch)
                        successful_items += len(batch)
                    except Exception as e:
                        failed_items += len(batch)
                        logger.error(f"Batch failed: {e}")
                
                benchmark_results.append(self.benchmark_history[-1])
                
                # Log progress
                if (i // batch_size + 1) % 10 == 0:
                    progress = (i + len(batch)) / len(test_data) * 100
                    logger.info(f"Load test progress: {progress:.1f}%")
        
        finally:
            monitoring_active = False
            monitor_thread.join(timeout=2)
        
        # Calculate results
        total_time = time.time() - start_time
        total_items = successful_items + failed_items
        average_time_per_item = total_time / total_items if total_items > 0 else 0
        throughput = successful_items / total_time if total_time > 0 else 0
        error_rate = failed_items / total_items if total_items > 0 else 0
        cpu_average = statistics.mean(cpu_usage_samples) if cpu_usage_samples else 0
        
        result = LoadTestResult(
            test_name=test_name,
            total_items=total_items,
            successful_items=successful_items,
            failed_items=failed_items,
            total_time=total_time,
            average_time_per_item=average_time_per_item,
            throughput_items_per_second=throughput,
            memory_usage_peak=memory_peak,
            cpu_usage_average=cpu_average,
            error_rate=error_rate,
            benchmark_results=benchmark_results
        )
        
        logger.info(f"Load test completed: {successful_items}/{total_items} successful, "
                   f"throughput: {throughput:.2f} items/s, error rate: {error_rate:.2%}")
        
        return result
    
    def generate_performance_report(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Performance report dictionary
        """
        if not self.benchmark_history:
            logger.warning("No benchmark data available for report")
            return {}
        
        # Analyze benchmark history
        operations = {}
        for result in self.benchmark_history:
            op_name = result.operation_name
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(result)
        
        # Generate statistics for each operation
        operation_stats = {}
        for op_name, results in operations.items():
            execution_times = [r.execution_time for r in results]
            memory_usage = [r.memory_peak.memory_percent for r in results]
            throughputs = [r.throughput for r in results if r.throughput]
            
            operation_stats[op_name] = {
                'count': len(results),
                'avg_execution_time': statistics.mean(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'avg_memory_usage': statistics.mean(memory_usage),
                'max_memory_usage': max(memory_usage),
                'avg_throughput': statistics.mean(throughputs) if throughputs else None,
                'max_throughput': max(throughputs) if throughputs else None
            }
        
        # System information
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate report
        report = {
            'system_info': system_info,
            'operation_statistics': operation_stats,
            'total_benchmarks': len(self.benchmark_history),
            'benchmark_period': {
                'start': min(r.timestamp for r in self.benchmark_history).isoformat(),
                'end': max(r.timestamp for r in self.benchmark_history).isoformat()
            }
        }
        
        # Save to file if requested
        if output_file:
            try:
                import json
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Performance report saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save performance report: {e}")
        
        return report
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations based on benchmark data"""
        recommendations = []
        
        if not self.benchmark_history:
            return ["No benchmark data available for recommendations"]
        
        # Analyze recent benchmarks
        recent_results = self.benchmark_history[-20:] if len(self.benchmark_history) > 20 else self.benchmark_history
        
        # Memory usage analysis
        high_memory_ops = [r for r in recent_results if r.memory_peak.memory_percent > 80]
        if high_memory_ops:
            recommendations.append(f"High memory usage detected in {len(high_memory_ops)} operations. Consider reducing batch sizes.")
        
        # Execution time analysis
        slow_ops = [r for r in recent_results if r.execution_time > 10]  # > 10 seconds
        if slow_ops:
            recommendations.append(f"Slow operations detected ({len(slow_ops)} ops > 10s). Consider optimization or caching.")
        
        # Throughput analysis
        throughput_ops = [r for r in recent_results if r.throughput and r.throughput < 1]  # < 1 item/second
        if throughput_ops:
            recommendations.append(f"Low throughput detected in {len(throughput_ops)} operations. Consider batch processing.")
        
        # Error analysis
        error_ops = [r for r in recent_results if r.error_count > 0]
        if error_ops:
            recommendations.append(f"Errors detected in {len(error_ops)} operations. Check error handling and data validation.")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable parameters.")
        
        return recommendations
    
    def clear_benchmark_history(self):
        """Clear benchmark history to free memory"""
        self.benchmark_history.clear()
        logger.info("Benchmark history cleared")