"""
Detection pipeline integration for applying direction stabilization to enemy submarines.
"""

from typing import Dict, List
from .direction_stabilizer import EnemySubmarineDirectionStabilizer


class SeaquestDetectionPipeline:
    """
    Detection pipeline that applies direction stabilization to enemy submarines.
    Uses visual detection (same as player) with frequency-based stabilization over 5 frames.
    """
    
    def __init__(self):
        """Initialize the detection pipeline with direction stabilizer."""
        self.direction_stabilizer = EnemySubmarineDirectionStabilizer(history_size=5)
    
    def process_detected_objects(self, detected_objects: Dict[str, List]) -> Dict[str, List]:
        """
        Process detected objects to apply direction stabilization to enemy submarines.
        
        Args:
            detected_objects: Dictionary mapping object types to lists of GameObjects
            
        Returns:
            Processed detected objects with stabilized directions
        """
        # Apply direction stabilization to enemy submarines
        enemy_submarines = detected_objects.get('enemy_submarine', [])
        for submarine in enemy_submarines:
            self.direction_stabilizer.update_submarine_direction(submarine)
        
        return detected_objects
    
    def get_debug_info(self) -> Dict:
        """
        Get debug information about the pipeline state.
        
        Returns:
            Dictionary with debug information
        """
        return {
            'pipeline_type': 'SeaquestDetectionPipeline',
            'direction_stabilizer_enabled': True,
            'stabilization_method': 'visual_frequency_based'
        }
    
    def reset(self):
        """Reset the pipeline state."""
        # Reset direction stabilizer
        for submarine_id in list(self.direction_stabilizer.stabilizer.direction_history.keys()):
            self.direction_stabilizer.stabilizer.reset_object(submarine_id)
