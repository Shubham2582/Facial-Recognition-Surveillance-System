"""System performance monitoring."""

import time
import psutil
import numpy as np
import logging
from typing import Dict, Any, List, Deque
from collections import deque

class PerformanceMonitor:
    """System performance monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance monitor.
        
        Args:
            config: Monitor configuration
        """
        self.config = config
        self.window_size = config.get('window_size', 100)
        
        # Initialize metrics
        self.metrics = {
            'detection_time': deque(maxlen=self.window_size),
            'tracking_time': deque(maxlen=self.window_size),
            'recognition_time': deque(maxlen=self.window_size),
            'total_processing_time': deque(maxlen=self.window_size),
            'fps': deque(maxlen=self.window_size),
            'face_count': deque(maxlen=self.window_size),
            'memory_usage': deque(maxlen=self.window_size),
            'cpu_usage': deque(maxlen=self.window_size),
            'gpu_memory_usage': deque(maxlen=self.window_size)
        }
        
        # Last update time
        self.last_update_time = time.time()
        
        # Frame counter
        self.frame_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize with empty metrics
        self.update_system_metrics()
        
    def log_metric(self, metric_name: str, value: float) -> None:
        """Log metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            
    def log_frame_stats(self, processing_time: float, face_count: int) -> None:
        """Log frame processing statistics.
        
        Args:
            processing_time: Frame processing time in seconds
            face_count: Number of faces detected
        """
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.log_metric('fps', fps)
            
            self.frame_count = 0
            self.last_update_time = current_time
            
        # Log metrics
        self.log_metric('total_processing_time', processing_time)
        self.log_metric('face_count', face_count)
        
        # Update system metrics periodically
        if len(self.metrics['memory_usage']) == 0 or elapsed >= 5.0:
            self.update_system_metrics()
            
    def update_system_metrics(self) -> None:
        """Update system performance metrics."""
        try:
            # Memory usage
            memory_info = psutil.Process().memory_info()
            memory_usage = memory_info.rss / (1024 * 1024)  # MB
            self.log_metric('memory_usage', memory_usage)
            
            # CPU usage
            cpu_usage = psutil.cpu_percent()
            self.log_metric('cpu_usage', cpu_usage)
            
            # GPU memory usage (placeholder - would need GPU-specific library)
            self.log_metric('gpu_memory_usage', 0.0)
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
            
    def get_avg_metric(self, metric_name: str, window: int = -1) -> float:
        """Get average metric value.
        
        Args:
            metric_name: Metric name
            window: Number of recent values to average (-1 for all)
            
        Returns:
            Average metric value
        """
        if metric_name in self.metrics:
            values = list(self.metrics[metric_name])
            if not values:
                return 0.0
                
            if window > 0:
                values = values[-min(window, len(values)):]
                
            return float(np.mean(values))
        return 0.0
        
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get detailed statistics for metric.
        
        Args:
            metric_name: Metric name
            
        Returns:
            Dictionary of metric statistics
        """
        if metric_name in self.metrics:
            values = list(self.metrics[metric_name])
            if not values:
                return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
                
            return {
                'avg': float(np.mean(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'std': float(np.std(values))
            }
        return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
    def print_performance_summary(self) -> None:
        """Print summary of performance metrics."""
        print("=== Performance Summary ===")
        print(f"Average FPS: {self.get_avg_metric('fps'):.2f}")
        print(f"Detection time: {self.get_avg_metric('detection_time')*1000:.2f} ms")
        print(f"Recognition time: {self.get_avg_metric('recognition_time')*1000:.2f} ms")
        print(f"Total processing: {self.get_avg_metric('total_processing_time')*1000:.2f} ms")
        print(f"Memory usage: {self.get_avg_metric('memory_usage'):.2f} MB")
        print(f"CPU usage: {self.get_avg_metric('cpu_usage'):.2f}%")
        print("===========================")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get all performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {}
        
        for metric_name in self.metrics:
            stats[metric_name] = self.get_metric_stats(metric_name)
            
        return stats