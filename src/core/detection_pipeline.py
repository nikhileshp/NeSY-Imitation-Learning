"""
Detection pipeline integration for applying direction stabilization to enemy submarines.
"""

from typing import Dict, List
from .simple_submarine_direction import SimpleSubmarineDirectionDetector


class SeaquestDetectionPipeline:
    """
    Detection pipeline that applies simple direction detection to enemy submarines.
    Uses x-coordinate of first detection frame to determine fixed facing direction.
    """
    
    def __init__(self):
        """Initialize the detection pipeline with simple direction detector."""
        self.direction_detector = SimpleSubmarineDirectionDetector()
    
    def process_detected_objects(self, detected_objects: Dict[str, List]) -> Dict[str, List]:
        """
        Process detected objects to apply simple direction detection to enemy submarines.
        
        Args:
            detected_objects: Dictionary mapping object types to lists of GameObjects
            
        Returns:
            Processed detected objects with direction detection applied
        """
        # Apply simple direction detection to enemy submarines
        enemy_submarines = detected_objects.get('enemy_submarine', [])
        for submarine in enemy_submarines:
            self.direction_detector.detect_submarine_direction(submarine)
        
        return detected_objects
    
    def get_debug_info(self) -> Dict:
        """
        Get debug information about the pipeline state.
        
        Returns:
            Dictionary with debug information
        """
        return {
            'pipeline_type': 'SeaquestDetectionPipeline',
            'direction_detector_enabled': True,
            'detection_method': 'x_coordinate_first_frame',
            'detector_debug': self.direction_detector.get_debug_info()
        }
    
    def reset(self):
        """Reset the pipeline state."""
        # Reset direction detector
        self.direction_detector.reset_all()
