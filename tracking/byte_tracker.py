"""Lightweight tracking implementation based on ByteTrack."""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
import logging

class ByteTracker:
    """Lightweight tracker based on ByteTrack algorithm."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ByteTracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config
        self.track_threshold = config.get('track_threshold', 0.5)
        self.track_buffer = config.get('track_buffer', 30)
        self.match_threshold = config.get('match_threshold', 0.8)
        self.frame_rate = config.get('frame_rate', 30)
        
        # Initialize trackers
        self.tracked_tracks = []
        self.lost_tracks = []
        self.next_track_id = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def update(self, detections: List[Dict[str, Any]], 
             frame: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracking with new detections.
        
        Args:
            detections: List of detections
            frame: Current frame
            
        Returns:
            List of tracked detections
        """
        # Convert detections to tracking format
        detection_boxes = []
        for det in detections:
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                detection_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': det.get('det_score', 1.0),
                    'detection': det
                })
        
        # Match with existing tracks
        tracked_boxes = self._match_tracks(detection_boxes, frame)
        
        # Convert back to original format with tracking info
        tracked_detections = []
        for box in tracked_boxes:
            if 'detection' in box:
                det = box['detection']
                det['track_id'] = box.get('track_id', -1)
                det['tracking_score'] = box.get('tracking_score', 0.0)
                
                # Update with trajectory if available
                if 'trajectory' in box:
                    det['trajectory'] = box['trajectory']
                    
                tracked_detections.append(det)
        
        return tracked_detections
        
    def _match_tracks(self, detections: List[Dict[str, Any]], 
                    frame: np.ndarray) -> List[Dict[str, Any]]:
        """Match detections with existing tracks.
        
        Args:
            detections: List of detections
            frame: Current frame
            
        Returns:
            List of tracked boxes
        """
        # Predict track locations
        for track in self.tracked_tracks + self.lost_tracks:
            if 'kalman' in track:
                # Predict next location
                track['predicted_bbox'] = self._kalman_predict(track['kalman'])
            else:
                # Use last bbox
                track['predicted_bbox'] = track['bbox']
        
        # Match high-confidence detections with tracked tracks
        high_dets = [d for d in detections if d['score'] >= self.track_threshold]
        matched_track_indices = []
        matched_det_indices = []
        
        if high_dets and self.tracked_tracks:
            # Compute IoU matrix
            iou_matrix = self._compute_iou_matrix(
                [t['predicted_bbox'] for t in self.tracked_tracks],
                [d['bbox'] for d in high_dets]
            )
            
            # Apply Hungarian algorithm or greedy matching
            matches = self._greedy_match(iou_matrix, self.match_threshold)
            
            for track_idx, det_idx in matches:
                # Update track with detection
                track = self.tracked_tracks[track_idx]
                det = high_dets[det_idx]
                
                # Update track info
                track['bbox'] = det['bbox']
                track['score'] = det['score']
                track['last_seen'] = 0
                
                # Update Kalman filter
                if 'kalman' in track:
                    self._kalman_update(track['kalman'], det['bbox'])
                else:
                    track['kalman'] = self._init_kalman(det['bbox'])
                    
                # Add tracking info to detection
                det['track_id'] = track['track_id']
                det['tracking_score'] = 1.0
                
                # Add trajectory info
                if 'trajectory' not in track:
                    track['trajectory'] = []
                    
                center_x = (det['bbox'][0] + det['bbox'][2]) / 2
                center_y = (det['bbox'][1] + det['bbox'][3]) / 2
                track['trajectory'].append((center_x, center_y))
                
                # Limit trajectory length
                if len(track['trajectory']) > 30:
                    track['trajectory'] = track['trajectory'][-30:]
                    
                det['trajectory'] = track['trajectory']
                
                # Mark as matched
                matched_track_indices.append(track_idx)
                matched_det_indices.append(det_idx)
        
        # Update unmatched tracks
        unmatched_track_indices = [i for i in range(len(self.tracked_tracks)) 
                                 if i not in matched_track_indices]
                                 
        for idx in unmatched_track_indices:
            track = self.tracked_tracks[idx]
            track['last_seen'] += 1
            
            # Mark as lost if not seen for too long
            if track['last_seen'] > self.track_buffer:
                self.lost_tracks.append(track)
                continue
        
        # Remove lost tracks
        self.tracked_tracks = [self.tracked_tracks[i] for i in range(len(self.tracked_tracks))
                            if i in matched_track_indices or 
                            self.tracked_tracks[i]['last_seen'] <= self.track_buffer]
        
        # Create new tracks for unmatched detections
        unmatched_det_indices = [i for i in range(len(high_dets)) 
                               if i not in matched_det_indices]
                               
        for idx in unmatched_det_indices:
            det = high_dets[idx]
            
            # Create new track
            track = {
                'track_id': self.next_track_id,
                'bbox': det['bbox'],
                'score': det['score'],
                'last_seen': 0,
                'kalman': self._init_kalman(det['bbox']),
                'trajectory': []
            }
            
            # Add trajectory
            center_x = (det['bbox'][0] + det['bbox'][2]) / 2
            center_y = (det['bbox'][1] + det['bbox'][3]) / 2
            track['trajectory'].append((center_x, center_y))
            
            # Add tracking info to detection
            det['track_id'] = track['track_id']
            det['tracking_score'] = 1.0
            det['trajectory'] = track['trajectory']
            
            # Increment track ID
            self.next_track_id += 1
            
            # Add to tracked tracks
            self.tracked_tracks.append(track)
        
        # Handle low-confidence detections
        low_dets = [d for d in detections if d['score'] < self.track_threshold]
        
        # Match low-confidence detections with lost tracks
        if low_dets and self.lost_tracks:
            # Compute IoU matrix
            iou_matrix = self._compute_iou_matrix(
                [t['predicted_bbox'] for t in self.lost_tracks],
                [d['bbox'] for d in low_dets]
            )
            
            # Apply matching
            matches = self._greedy_match(iou_matrix, self.match_threshold)
            
            for track_idx, det_idx in matches:
                # Recover lost track
                track = self.lost_tracks[track_idx]
                det = low_dets[det_idx]
                
                # Update track info
                track['bbox'] = det['bbox']
                track['score'] = det['score']
                track['last_seen'] = 0
                
                # Update Kalman filter
                if 'kalman' in track:
                    self._kalman_update(track['kalman'], det['bbox'])
                    
                # Add tracking info to detection
                det['track_id'] = track['track_id']
                det['tracking_score'] = 0.5
                
                # Update trajectory
                if 'trajectory' not in track:
                    track['trajectory'] = []
                    
                center_x = (det['bbox'][0] + det['bbox'][2]) / 2
                center_y = (det['bbox'][1] + det['bbox'][3]) / 2
                track['trajectory'].append((center_x, center_y))
                
                # Limit trajectory length
                if len(track['trajectory']) > 30:
                    track['trajectory'] = track['trajectory'][-30:]
                    
                det['trajectory'] = track['trajectory']
                
                # Move back to tracked tracks
                self.tracked_tracks.append(track)
        
        # Remove recovered tracks from lost tracks
        self.lost_tracks = [track for track in self.lost_tracks 
                          if track not in self.tracked_tracks]
        
        # Remove old lost tracks
        self.lost_tracks = [track for track in self.lost_tracks 
                          if track['last_seen'] <= self.track_buffer * 2]
        
        # Combine all tracked detections
        result = []
        for det in high_dets + low_dets:
            if 'track_id' in det:
                result.append(det)
                
        return result
        
    def _compute_iou_matrix(self, boxes1: List[List[float]], 
                          boxes2: List[List[float]]) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes.
        
        Args:
            boxes1: First set of boxes (N, 4)
            boxes2: Second set of boxes (M, 4)
            
        Returns:
            IoU matrix (N, M)
        """
        n = len(boxes1)
        m = len(boxes2)
        iou_matrix = np.zeros((n, m), dtype=np.float32)
        
        for i in range(n):
            for j in range(m):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])
                
        return iou_matrix
        
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Check if boxes overlap
        if x2 < x1 or y2 < y1:
            return 0.0
            
        # Compute intersection area
        intersection = (x2 - x1) * (y2 - y1)
        
        # Compute union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # Compute IoU
        if union > 0:
            return intersection / union
        else:
            return 0.0
            
    def _greedy_match(self, cost_matrix: np.ndarray, 
                    threshold: float) -> List[Tuple[int, int]]:
        """Perform greedy matching based on cost matrix.
        
        Args:
            cost_matrix: Cost matrix
            threshold: Matching threshold
            
        Returns:
            List of matched indices (row_idx, col_idx)
        """
        # Make a copy of the cost matrix
        costs = cost_matrix.copy()
        
        # Get matrix dimensions
        n, m = costs.shape
        
        # Initialize matched indices
        matches = []
        
        # Greedily select pairs with highest cost
        while costs.size > 0 and np.max(costs) >= threshold:
            # Get highest cost indices
            row, col = np.unravel_index(np.argmax(costs), costs.shape)
            
            # Add to matches
            matches.append((row, col))
            
            # Remove matched elements
            costs = np.delete(costs, row, axis=0)
            if costs.size > 0:
                costs = np.delete(costs, col, axis=1)
                
        # Convert to original indices
        original_matches = []
        row_indices = list(range(n))
        col_indices = list(range(m))
        
        for match in matches:
            row_idx = row_indices[match[0]]
            col_idx = col_indices[match[1]]
            original_matches.append((row_idx, col_idx))
            
            # Update remaining indices
            row_indices.pop(match[0])
            col_indices.pop(match[1])
            
        return original_matches
        
    def _init_kalman(self, bbox: List[float]) -> Dict[str, Any]:
        """Initialize Kalman filter for tracking.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Kalman filter state
        """
        # Extract box parameters
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Initialize state [x, y, w, h, vx, vy, vw, vh]
        state = np.array([center_x, center_y, width, height, 0, 0, 0, 0], dtype=np.float32)
        
        # Initialize transition matrix
        # [1, 0, 0, 0, dt, 0, 0, 0]
        # [0, 1, 0, 0, 0, dt, 0, 0]
        # [0, 0, 1, 0, 0, 0, dt, 0]
        # [0, 0, 0, 1, 0, 0, 0, dt]
        # [0, 0, 0, 0, 1, 0, 0, 0]
        # [0, 0, 0, 0, 0, 1, 0, 0]
        # [0, 0, 0, 0, 0, 0, 1, 0]
        # [0, 0, 0, 0, 0, 0, 0, 1]
        transition_matrix = np.eye(8, dtype=np.float32)
        transition_matrix[0, 4] = 1.0 / self.frame_rate
        transition_matrix[1, 5] = 1.0 / self.frame_rate
        transition_matrix[2, 6] = 1.0 / self.frame_rate
        transition_matrix[3, 7] = 1.0 / self.frame_rate
        
        # Initialize measurement matrix
        # [1, 0, 0, 0, 0, 0, 0, 0]
        # [0, 1, 0, 0, 0, 0, 0, 0]
        # [0, 0, 1, 0, 0, 0, 0, 0]
        # [0, 0, 0, 1, 0, 0, 0, 0]
        measurement_matrix = np.zeros((4, 8), dtype=np.float32)
        measurement_matrix[0, 0] = 1.0
        measurement_matrix[1, 1] = 1.0
        measurement_matrix[2, 2] = 1.0
        measurement_matrix[3, 3] = 1.0
        
        # Initialize covariance matrix
        covariance = np.eye(8, dtype=np.float32) * 100.0
        
        # Initialize process noise covariance
        process_noise = np.eye(8, dtype=np.float32) * 0.01
        
        # Initialize measurement noise covariance
        measurement_noise = np.eye(4, dtype=np.float32) * 1.0
        
        return {
            'state': state,
            'transition_matrix': transition_matrix,
            'measurement_matrix': measurement_matrix,
            'covariance': covariance,
            'process_noise': process_noise,
            'measurement_noise': measurement_noise
        }
        
    def _kalman_predict(self, kalman: Dict[str, Any]) -> List[float]:
        """Predict next state using Kalman filter.
        
        Args:
            kalman: Kalman filter state
            
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Predict next state
        state = kalman['state']
        transition_matrix = kalman['transition_matrix']
        covariance = kalman['covariance']
        process_noise = kalman['process_noise']
        
        # x' = Fx
        state = np.dot(transition_matrix, state)
        
        # P' = FPF' + Q
        covariance = np.dot(np.dot(transition_matrix, covariance), transition_matrix.T) + process_noise
        
        # Update Kalman state
        kalman['state'] = state
        kalman['covariance'] = covariance
        
        # Convert to bounding box
        center_x, center_y, width, height = state[:4]
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        return [x1, y1, x2, y2]
        
    def _kalman_update(self, kalman: Dict[str, Any], 
                      bbox: List[float]) -> None:
        """Update Kalman filter with measurement.
        
        Args:
            kalman: Kalman filter state
            bbox: Measured bounding box [x1, y1, x2, y2]
        """
        # Extract box parameters
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Create measurement
        measurement = np.array([center_x, center_y, width, height], dtype=np.float32)
        
        # Get Kalman state
        state = kalman['state']
        covariance = kalman['covariance']
        measurement_matrix = kalman['measurement_matrix']
        measurement_noise = kalman['measurement_noise']
        
        # Calculate Kalman gain
        # K = PH'(HPH' + R)^-1
        kalman_gain_numerator = np.dot(covariance, measurement_matrix.T)
        kalman_gain_denominator = np.dot(np.dot(measurement_matrix, covariance), measurement_matrix.T) + measurement_noise
        kalman_gain = np.dot(kalman_gain_numerator, np.linalg.inv(kalman_gain_denominator))
        
        # Update state
        # x = x' + K(z - Hx')
        innovation = measurement - np.dot(measurement_matrix, state)
        state = state + np.dot(kalman_gain, innovation)
        
        # Update covariance
        # P = (I - KH)P'
        identity = np.eye(8, dtype=np.float32)
        covariance = np.dot(identity - np.dot(kalman_gain, measurement_matrix), covariance)
        
        # Update Kalman state
        kalman['state'] = state
        kalman['covariance'] = covariance