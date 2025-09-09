"""
Simple submarine direction detector based on x-coordinate of first detection frame.
High x value (right side of screen) = moving left
Low x value (left side of screen) = moving right
Direction is fixed once determined for each submarine.
"""

from typing import Dict, Optional


class SimpleSubmarineDirectionDetector:
    """
    Simple submarine direction detector that determines facing direction based on 
    the x-coordinate of the first frame where each submarine is detected.
    """
    
    def __init__(self, x_threshold: int = 80):
        """
        Initialize the simple submarine direction detector.
        
        Args:
            x_threshold: X-coordinate threshold to distinguish left/right
                        positions (default 80, center of 160px wide Seaquest screen)
        """
        self.x_threshold = x_threshold
    
    def detect_submarine_direction(self, submarine_object) -> Optional[str]:
        """
        Detect and set the facing direction for a submarine based on its x-coordinate.
        Only sets direction if the submarine doesn't already have one.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
            
        Returns:
            The facing direction ('left' or 'right') or None if not a submarine
        """
        if submarine_object.object_type != 'enemy_submarine':
            return None
        
        # Check if submarine already has a facing direction
        if (hasattr(submarine_object, 'characteristics') and 
            'facing_side' in submarine_object.characteristics):
            # Return existing direction
            return submarine_object.characteristics['facing_side']
        
        # First time seeing this submarine - determine direction from x-coordinate
        x_position = submarine_object.x
        
        if x_position > self.x_threshold:
            # High x value (right side of screen) = moving left
            direction = 'left'
        else:
            # Low x value (left side of screen) = moving right
            direction = 'right'
        
        # Set the submarine object's characteristics
        if not hasattr(submarine_object, 'characteristics'):
            submarine_object.characteristics = {}

        submarine_object.characteristics['facing_side'] = direction
        submarine_object.characteristics['facing_source'] = 'x_coordinate_first_frame'

        return direction
    
    def get_debug_info(self) -> Dict:
        """
        Get debug information about the detector.
        
        Returns:
            Dictionary with debug information
        """
        return {
            'detector_type': 'SimpleSubmarineDirectionDetector',
            'x_threshold': self.x_threshold,
            'detection_method': 'x_coordinate_first_frame_only'
        }
    
    def set_x_threshold(self, threshold: int):
        """
        Set the x-coordinate threshold for direction determination.
        
        Args:
            threshold: New x-coordinate threshold
        """
        self.x_threshold = threshold
