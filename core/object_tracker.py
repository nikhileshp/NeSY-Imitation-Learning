"""
Object tracking system for maintaining consistent indexes across frames.
Uses OC-Atari's Hungarian algorithm-based matching approach.
"""
import numpy as np
from typing import Dict, List, Optional, Any
from scipy.optimize import linear_sum_assignment

from core.game_object import GameObject


class NoObject:
    """Represents a non-existent object slot in the tracking system."""
    
    def __init__(self):
        self._xy = (0, 0)
        self.wh = (0, 0)
        self._visible = False
        
    def __bool__(self):
        return False
        
    def __repr__(self):
        return "NoObject"


class TrackableGameObject(GameObject):
    """Extended GameObject with tracking capabilities for OC-Atari compatibility."""
    
    def __init__(self, object_type: str, bounding_box: tuple, object_id: str = None, 
                 characteristics: dict = None):
        super().__init__(object_type, bounding_box, object_id, characteristics)
        # Add tracking properties required by OC-Atari functions
        self._xy = (self.x, self.y)
        self.wh = (self.width, self.height)
        self.num_frames_invisible = 0
        self.max_frames_invisible = 5  # Allow object to be invisible for up to 5 frames
        self.expected_dist = 50  # Maximum expected movement distance between frames
        
    @property
    def xywh(self):
        """Return (x, y, w, h) tuple for OC-Atari compatibility."""
        return (self.x, self.y, self.width, self.height)
        
    @xywh.setter
    def xywh(self, xywh):
        """Set position and size from (x, y, w, h) tuple."""
        self.x, self.y, self.width, self.height = xywh
        self._xy = (self.x, self.y)
        self.wh = (self.width, self.height)


class ObjectTracker:
    """
    Object tracker that maintains consistent indexes across frames using 
    OC-Atari's Hungarian algorithm-based matching approach.
    """
    
    def __init__(self, max_objects_per_type: Dict[str, int] = None):
        """
        Initialize the object tracker.
        
        Args:
            max_objects_per_type: Maximum number of objects allowed for each type
        """
        self.max_objects_per_type = max_objects_per_type or {}
        self.previous_objects = {}  # Track objects from previous frame
        self.current_frame = 0
        
    def _compute_cost_matrix(self, prev_objects: List, current_bboxes: List) -> np.ndarray:
        """
        Compute cost matrix between previous objects and current detections.
        Uses L1 distance between object centers.
        """
        cost_matrix = np.zeros((len(prev_objects), len(current_bboxes)))
        
        for i, prev_obj in enumerate(prev_objects):
            for j, curr_bbox in enumerate(current_bboxes):
                if not prev_obj or isinstance(prev_obj, NoObject):
                    cost_matrix[i, j] = 1000  # Large value for no previous object
                else:
                    # L1 distance between centers
                    prev_center = np.array([prev_obj.x + prev_obj.width//2, 
                                          prev_obj.y + prev_obj.height//2])
                    curr_center = np.array([curr_bbox[0] + curr_bbox[2]//2, 
                                          curr_bbox[1] + curr_bbox[3]//2])
                    cost_matrix[i, j] = np.sum(np.abs(prev_center - curr_center))
                    
        return cost_matrix
    
    def _convert_to_bounding_boxes(self, objects: List[GameObject]) -> List[tuple]:
        """Convert GameObject list to bounding box tuples for matching."""
        return [obj.bounding_box for obj in objects]
    
    def _create_trackable_object(self, object_type: str, bbox: tuple, index: int) -> TrackableGameObject:
        """Create a new trackable object with consistent ID."""
        object_id = f"{object_type}_{index}"
        obj = TrackableGameObject(object_type, bbox, object_id)
        # Ensure the object_id is set correctly
        obj.object_id = object_id
        return obj
    
    def match_objects_for_type(self, object_type: str, detected_objects: List[GameObject]) -> List[GameObject]:
        """
        Match detected objects with previous frame objects to maintain consistent indexes.
        
        Args:
            object_type: Type of objects to match (e.g., 'enemy', 'player')
            detected_objects: List of newly detected objects
            
        Returns:
            List of matched objects with consistent IDs
        """
        max_objects = self.max_objects_per_type.get(object_type, len(detected_objects) + 10)
        
        # Initialize previous objects list if first frame
        if object_type not in self.previous_objects:
            self.previous_objects[object_type] = [NoObject() for _ in range(max_objects)]
        
        prev_objects = self.previous_objects[object_type]
        current_bboxes = self._convert_to_bounding_boxes(detected_objects)
        
        # If no current detections, mark previous objects as potentially invisible
        if len(current_bboxes) == 0:
            for obj in prev_objects:
                if obj and not isinstance(obj, NoObject):
                    obj.num_frames_invisible += 1
                    if obj.num_frames_invisible > obj.max_frames_invisible:
                        # Replace with NoObject if invisible too long
                        idx = prev_objects.index(obj)
                        prev_objects[idx] = NoObject()
            return [obj for obj in prev_objects if obj and not isinstance(obj, NoObject)]
        
        # If no previous objects, create new ones with consistent IDs
        if all(isinstance(obj, NoObject) or not obj for obj in prev_objects):
            for i in range(min(max_objects, len(current_bboxes))):
                prev_objects[i] = self._create_trackable_object(object_type, current_bboxes[i], i)
            return [obj for obj in prev_objects if obj and not isinstance(obj, NoObject)]
        
        # Perform Hungarian matching
        cost_matrix = self._compute_cost_matrix(prev_objects, current_bboxes)
        obj_indices, bbox_indices = linear_sum_assignment(cost_matrix)
        
        # Create new object list for this frame
        new_objects = [NoObject() for _ in range(max_objects)]
        
        # Update matched objects
        for obj_idx, bbox_idx in zip(obj_indices, bbox_indices):
            if obj_idx < len(prev_objects) and bbox_idx < len(current_bboxes):
                if isinstance(prev_objects[obj_idx], NoObject) or not prev_objects[obj_idx]:
                    # Create new object with the same index
                    new_objects[obj_idx] = self._create_trackable_object(
                        object_type, current_bboxes[bbox_idx], obj_idx)
                else:
                    # Update existing object
                    existing_obj = prev_objects[obj_idx]
                    existing_obj.x, existing_obj.y, existing_obj.width, existing_obj.height = current_bboxes[bbox_idx]
                    existing_obj._xy = (existing_obj.x, existing_obj.y)
                    existing_obj.wh = (existing_obj.width, existing_obj.height)
                    existing_obj.num_frames_invisible = 0
                    new_objects[obj_idx] = existing_obj
        
        # Handle unmatched detections - assign to first available slots
        matched_bbox_indices = set(bbox_indices)
        available_slots = [i for i, obj in enumerate(new_objects) 
                          if isinstance(obj, NoObject) or not obj]
        
        unmatched_detections = [i for i in range(len(current_bboxes)) 
                               if i not in matched_bbox_indices]
        
        for bbox_idx in unmatched_detections:
            if available_slots:
                slot_idx = available_slots.pop(0)
                new_objects[slot_idx] = self._create_trackable_object(
                    object_type, current_bboxes[bbox_idx], slot_idx)
        
        # Update previous objects for next frame
        self.previous_objects[object_type] = new_objects
        
        # Return only non-empty objects
        return [obj for obj in new_objects if obj and not isinstance(obj, NoObject)]
    
    def track_all_objects(self, detected_objects: Dict[str, List[GameObject]]) -> Dict[str, List[GameObject]]:
        """
        Track all object types and maintain consistent indexes.
        
        Args:
            detected_objects: Dictionary mapping object types to detected objects
            
        Returns:
            Dictionary with tracked objects maintaining consistent IDs
        """
        self.current_frame += 1
        tracked_objects = {}
        
        for object_type, objects in detected_objects.items():
            tracked_objects[object_type] = self.match_objects_for_type(object_type, objects)
            
        return tracked_objects
    
    def reset(self):
        """Reset the tracker state."""
        self.previous_objects = {}
        self.current_frame = 0
    
    def set_max_objects(self, object_type: str, max_count: int):
        """Set maximum number of objects for a specific type."""
        self.max_objects_per_type[object_type] = max_count
    
    def get_tracking_info(self) -> Dict[str, Any]:
        """Get current tracking information for debugging."""
        info = {
            'current_frame': self.current_frame,
            'tracked_types': list(self.previous_objects.keys()),
            'max_objects': self.max_objects_per_type.copy()
        }
        
        for obj_type, objects in self.previous_objects.items():
            active_count = sum(1 for obj in objects if obj and not isinstance(obj, NoObject))
            info[f'{obj_type}_active_count'] = active_count
            
        return info
