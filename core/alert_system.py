"""Alert generation system for facial recognition surveillance."""

import time
import logging
import threading
import queue
from typing import Dict, Any, List, Optional, Callable

class AlertSystem:
    """Alert generation system for facial recognition surveillance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize alert system.
        
        Args:
            config: Alert system configuration
        """
        self.config = config
        self.alert_threshold = config.get('alert_threshold', 0.7)
        self.cooldown_period = config.get('cooldown_period', 60.0)  # seconds
        self.max_alerts = config.get('max_alerts', 100)
        
        # Initialize alert queue
        self.alert_queue = queue.Queue(maxsize=self.max_alerts)
        
        # Initialize alert callbacks
        self.alert_callbacks = []
        
        # Initialize cooldown tracking
        self.last_alerts = {}  # identity -> timestamp
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add alert callback function.
        
        Args:
            callback: Callback function
        """
        self.alert_callbacks.append(callback)
        
    def check_alert(self, detection: Dict[str, Any]) -> bool:
        """Check if detection should trigger alert.
        
        Args:
            detection: Detection information
            
        Returns:
            True if alert should be generated
        """
        identity = detection.get('identity', 'Unknown')
        confidence = detection.get('confidence', 0.0)
        
        # Skip unknown identities
        if identity == 'Unknown':
            return False
            
        # Check confidence threshold
        if confidence < self.alert_threshold:
            return False
            
        # Check cooldown period
        now = time.time()
        if identity in self.last_alerts:
            last_time = self.last_alerts[identity]
            if now - last_time < self.cooldown_period:
                return False
                
        # Update last alert time
        self.last_alerts[identity] = now
        
        return True
        
    def generate_alert(self, detection: Dict[str, Any],
                      frame: Optional[Any] = None) -> Dict[str, Any]:
        """Generate alert for detection.
        
        Args:
            detection: Detection information
            frame: Frame image (optional)
            
        Returns:
            Alert information
        """
        # Create alert
        alert = {
            'timestamp': time.time(),
            'identity': detection.get('identity', 'Unknown'),
            'confidence': detection.get('confidence', 0.0),
            'bbox': detection.get('bbox', None),
            'location': detection.get('location', 'Unknown'),
            'camera_id': detection.get('camera_id', 'Unknown'),
            'frame': frame
        }
        
        # Log alert
        self.logger.info(f"Alert generated: {alert['identity']} detected with confidence {alert['confidence']:.2f}")
        
        # Add to queue
        try:
            self.alert_queue.put(alert, block=False)
        except queue.Full:
            self.logger.warning("Alert queue full, discarding oldest alert")
            try:
                # Remove oldest alert
                self.alert_queue.get_nowait()
                self.alert_queue.put(alert, block=False)
            except Exception:
                pass
                
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
                
        return alert
        
    def process_detections(self, detections: List[Dict[str, Any]], 
                          frame: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Process detections and generate alerts.
        
        Args:
            detections: List of detection information
            frame: Frame image (optional)
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        for detection in detections:
            if self.check_alert(detection):
                alert = self.generate_alert(detection, frame)
                alerts.append(alert)
                
        return alerts
        
    def get_alerts(self, max_alerts: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts.
        
        Args:
            max_alerts: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        alerts = []
        
        # Get alerts from queue without removing them
        queue_size = self.alert_queue.qsize()
        for _ in range(min(queue_size, max_alerts)):
            try:
                alert = self.alert_queue.get()
                alerts.append(alert)
                self.alert_queue.put(alert)
            except Exception:
                break
                
        return alerts