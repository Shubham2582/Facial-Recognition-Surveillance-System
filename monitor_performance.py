# monitor_performance.py
import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
from utils.performance_monitor import PerformanceMonitor
from config.system_config import SYSTEM_CONFIG

def main():
    # Initialize performance monitor
    monitor = PerformanceMonitor(SYSTEM_CONFIG)
    
    # Initialize plotting
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    x_data = []
    fps_data = []
    cpu_data = []
    memory_data = []
    
    # Mock processing for demonstration
    for i in range(100):
        # Update system metrics
        monitor.update_system_metrics()
        
        # Log mock metrics
        mock_processing_time = 0.03 + np.random.normal(0, 0.01)
        mock_face_count = np.random.randint(1, 5)
        monitor.log_frame_stats(mock_processing_time, mock_face_count)
        
        # Get metrics
        fps = monitor.get_avg_metric('fps')
        cpu = monitor.get_avg_metric('cpu_usage')
        memory = monitor.get_avg_metric('memory_usage')
        
        # Update plots
        x_data.append(i)
        fps_data.append(fps)
        cpu_data.append(cpu)
        memory_data.append(memory)
        
        ax1.clear()
        ax1.plot(x_data, fps_data)
        ax1.set_title('FPS')
        ax1.set_ylim(0, 30)
        
        ax2.clear()
        ax2.plot(x_data, cpu_data)
        ax2.set_title('CPU Usage (%)')
        ax2.set_ylim(0, 100)
        
        ax3.clear()
        ax3.plot(x_data, memory_data)
        ax3.set_title('Memory Usage (MB)')
        
        plt.tight_layout()
        plt.pause(0.1)
        
        # Print summary periodically
        if i % 10 == 0:
            monitor.print_performance_summary()
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()