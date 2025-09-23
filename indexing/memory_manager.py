"""
Memory management and optimization utilities for SecureInsight
"""
import gc
import mmap
import os
import psutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from loguru import logger
from config import PERFORMANCE_CONFIG


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    process_memory: float
    gc_collections: Dict[int, int]
    timestamp: datetime


class MemoryManager:
    """Advanced memory management for large dataset processing"""
    
    def __init__(self, gc_threshold: float = 0.8, monitoring_interval: int = 30):
        """
        Initialize memory manager
        
        Args:
            gc_threshold: Memory usage threshold to trigger garbage collection (0.0-1.0)
            monitoring_interval: Monitoring interval in seconds
        """
        self.gc_threshold = gc_threshold
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_thread = None
        self.memory_stats_history = []
        self.max_history_size = 1000
        
        # Configure garbage collection
        self._configure_gc()
        
        logger.info(f"Memory manager initialized with GC threshold: {gc_threshold}")
    
    def _configure_gc(self):
        """Configure garbage collection for optimal performance"""
        try:
            # Set more aggressive garbage collection thresholds
            # Default is (700, 10, 10), we make it more frequent
            gc.set_threshold(500, 8, 8)
            
            # Enable automatic garbage collection
            gc.enable()
            
            logger.info("Garbage collection configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure garbage collection: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Garbage collection stats
            gc_stats = {}
            for i in range(3):
                gc_stats[i] = gc.get_count()[i]
            
            return MemoryStats(
                total_memory=memory.total / (1024**3),  # GB
                available_memory=memory.available / (1024**3),  # GB
                used_memory=memory.used / (1024**3),  # GB
                memory_percent=memory.percent,
                process_memory=process_memory,
                gc_collections=gc_stats,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0, {}, datetime.now())
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        stats = self.get_memory_stats()
        return stats.memory_percent > (self.gc_threshold * 100)
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return collection counts"""
        try:
            # Get counts before collection
            before_counts = gc.get_count()
            
            # Force collection for all generations
            collected = {}
            for generation in range(3):
                collected[generation] = gc.collect(generation)
            
            # Get counts after collection
            after_counts = gc.get_count()
            
            logger.info(f"Forced GC: collected {sum(collected.values())} objects")
            
            return {
                'collected': collected,
                'before_counts': before_counts,
                'after_counts': after_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to force garbage collection: {e}")
            return {}
    
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                stats = self.get_memory_stats()
                
                # Add to history
                self.memory_stats_history.append(stats)
                
                # Limit history size
                if len(self.memory_stats_history) > self.max_history_size:
                    self.memory_stats_history.pop(0)
                
                # Check for memory pressure and trigger GC if needed
                if stats.memory_percent > (self.gc_threshold * 100):
                    logger.warning(f"Memory pressure detected: {stats.memory_percent:.1f}%")
                    self.force_gc()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_memory_history(self, last_n: Optional[int] = None) -> List[MemoryStats]:
        """Get memory usage history"""
        if last_n is None:
            return self.memory_stats_history.copy()
        return self.memory_stats_history[-last_n:] if self.memory_stats_history else []
    
    def optimize_for_large_dataset(self):
        """Optimize memory settings for large dataset processing"""
        try:
            # More aggressive garbage collection
            gc.set_threshold(300, 5, 5)
            
            # Force initial cleanup
            self.force_gc()
            
            logger.info("Memory optimized for large dataset processing")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory for large dataset: {e}")
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations based on current usage"""
        recommendations = []
        stats = self.get_memory_stats()
        
        if stats.memory_percent > 90:
            recommendations.append("Critical: Memory usage above 90%. Consider reducing batch size.")
        elif stats.memory_percent > 80:
            recommendations.append("Warning: High memory usage. Monitor closely.")
        
        if stats.process_memory > 2000:  # 2GB
            recommendations.append("Process using significant memory. Consider model quantization.")
        
        # Check GC frequency
        if len(self.memory_stats_history) > 10:
            recent_stats = self.memory_stats_history[-10:]
            gc_frequency = sum(s.gc_collections.get(0, 0) for s in recent_stats) / len(recent_stats)
            
            if gc_frequency > 100:
                recommendations.append("High GC frequency detected. Consider optimizing object creation.")
        
        if not recommendations:
            recommendations.append("Memory usage is within normal parameters.")
        
        return recommendations


class ProgressiveLoader:
    """Progressive loading for large datasets"""
    
    def __init__(self, chunk_size: int = 1000, memory_threshold: float = 0.8):
        """
        Initialize progressive loader
        
        Args:
            chunk_size: Number of items to load per chunk
            memory_threshold: Memory usage threshold to pause loading
        """
        self.chunk_size = chunk_size
        self.memory_threshold = memory_threshold
        self.memory_manager = MemoryManager()
        
    def load_documents_progressively(self, document_paths: List[Path]) -> Iterator[List[Path]]:
        """
        Load documents progressively in chunks
        
        Args:
            document_paths: List of document paths to load
            
        Yields:
            Chunks of document paths
        """
        total_docs = len(document_paths)
        processed = 0
        
        logger.info(f"Starting progressive loading of {total_docs} documents")
        
        for i in range(0, total_docs, self.chunk_size):
            # Check memory pressure before loading next chunk
            if self.memory_manager.check_memory_pressure():
                logger.warning("Memory pressure detected, forcing garbage collection")
                self.memory_manager.force_gc()
                
                # Wait a bit for memory to stabilize
                time.sleep(1)
            
            chunk = document_paths[i:i + self.chunk_size]
            processed += len(chunk)
            
            logger.info(f"Loading chunk {i//self.chunk_size + 1}: {processed}/{total_docs} documents")
            
            yield chunk
    
    def load_embeddings_progressively(self, embeddings_file: Path, 
                                    chunk_size: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Load embeddings progressively from file
        
        Args:
            embeddings_file: Path to embeddings file
            chunk_size: Number of embeddings per chunk
            
        Yields:
            Chunks of embeddings
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        try:
            # Load embeddings in chunks using memory mapping
            with open(embeddings_file, 'rb') as f:
                # Assume embeddings are stored as numpy array
                embeddings = np.load(f, mmap_mode='r')
                
                total_embeddings = len(embeddings)
                logger.info(f"Loading {total_embeddings} embeddings progressively")
                
                for i in range(0, total_embeddings, chunk_size):
                    # Check memory before loading
                    if self.memory_manager.check_memory_pressure():
                        self.memory_manager.force_gc()
                        time.sleep(0.5)
                    
                    chunk = embeddings[i:i + chunk_size]
                    logger.debug(f"Loaded embedding chunk: {i}-{i+len(chunk)}")
                    
                    yield chunk.copy()  # Copy to avoid memory mapping issues
                    
        except Exception as e:
            logger.error(f"Failed to load embeddings progressively: {e}")
            raise


class MemoryMappedIndex:
    """Memory-mapped file access for large indices"""
    
    def __init__(self, index_file: Path, mode: str = 'r'):
        """
        Initialize memory-mapped index
        
        Args:
            index_file: Path to index file
            mode: File access mode ('r', 'w', 'r+')
        """
        self.index_file = Path(index_file)
        self.mode = mode
        self.file_handle = None
        self.mmap_handle = None
        self.metadata = {}
        
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def open(self):
        """Open memory-mapped file"""
        try:
            # Ensure file exists for read mode
            if self.mode == 'r' and not self.index_file.exists():
                raise FileNotFoundError(f"Index file not found: {self.index_file}")
            
            # Open file
            file_mode = 'rb' if 'r' in self.mode else 'r+b'
            if self.mode == 'w':
                file_mode = 'wb'
            
            self.file_handle = open(self.index_file, file_mode)
            
            # Create memory map
            if self.mode == 'r':
                self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            elif self.mode == 'w':
                # For write mode, we'll handle this differently
                pass
            else:  # r+
                self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_WRITE)
            
            logger.info(f"Opened memory-mapped index: {self.index_file}")
            
        except Exception as e:
            logger.error(f"Failed to open memory-mapped index: {e}")
            self.close()
            raise
    
    def close(self):
        """Close memory-mapped file"""
        try:
            if self.mmap_handle:
                self.mmap_handle.close()
                self.mmap_handle = None
            
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
            
            logger.debug(f"Closed memory-mapped index: {self.index_file}")
            
        except Exception as e:
            logger.error(f"Error closing memory-mapped index: {e}")
    
    def read_chunk(self, offset: int, size: int) -> bytes:
        """Read a chunk of data from the memory-mapped file"""
        if not self.mmap_handle:
            raise RuntimeError("Memory-mapped file not open")
        
        try:
            self.mmap_handle.seek(offset)
            return self.mmap_handle.read(size)
        except Exception as e:
            logger.error(f"Failed to read chunk at offset {offset}: {e}")
            raise
    
    def write_chunk(self, offset: int, data: bytes):
        """Write a chunk of data to the memory-mapped file"""
        if not self.mmap_handle:
            raise RuntimeError("Memory-mapped file not open for writing")
        
        try:
            self.mmap_handle.seek(offset)
            self.mmap_handle.write(data)
            self.mmap_handle.flush()
        except Exception as e:
            logger.error(f"Failed to write chunk at offset {offset}: {e}")
            raise
    
    def get_size(self) -> int:
        """Get the size of the memory-mapped file"""
        if self.mmap_handle:
            return len(self.mmap_handle)
        elif self.file_handle:
            return os.path.getsize(self.index_file)
        else:
            return 0
    
    def create_index_file(self, data: bytes):
        """Create a new index file with data"""
        try:
            with open(self.index_file, 'wb') as f:
                f.write(data)
            
            logger.info(f"Created index file: {self.index_file} ({len(data)} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to create index file: {e}")
            raise


class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def optimize_numpy_arrays(arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize numpy arrays for memory efficiency"""
        optimized = []
        
        for arr in arrays:
            # Convert to most efficient dtype
            if arr.dtype == np.float64:
                # Check if we can use float32 without significant precision loss
                arr_f32 = arr.astype(np.float32)
                if np.allclose(arr, arr_f32, rtol=1e-6):
                    arr = arr_f32
            
            # Ensure array is contiguous for better cache performance
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            
            optimized.append(arr)
        
        return optimized
    
    @staticmethod
    def compress_embeddings(embeddings: np.ndarray, precision: str = 'float16') -> np.ndarray:
        """Compress embeddings to reduce memory usage"""
        try:
            if precision == 'float16':
                return embeddings.astype(np.float16)
            elif precision == 'int8':
                # Quantize to int8 (requires normalization)
                normalized = embeddings / np.max(np.abs(embeddings))
                return (normalized * 127).astype(np.int8)
            else:
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to compress embeddings: {e}")
            return embeddings
    
    @staticmethod
    def estimate_memory_usage(data_size: int, dtype: str = 'float32') -> float:
        """Estimate memory usage in MB for given data size and type"""
        dtype_sizes = {
            'float64': 8,
            'float32': 4,
            'float16': 2,
            'int64': 8,
            'int32': 4,
            'int16': 2,
            'int8': 1
        }
        
        bytes_per_element = dtype_sizes.get(dtype, 4)
        total_bytes = data_size * bytes_per_element
        return total_bytes / (1024 ** 2)  # Convert to MB