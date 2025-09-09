"""
Simple direction stabilizer for enemy submarines.
Uses the most frequent direction from the last 5 detected directions.
"""

from typing import Dict, Optional, List
from collections import deque, Counter


class DirectionStabilizer:
    """Stabilizes facing direction by tracking the most frequent direction over recent frames."""
    
    def __init__(self, history_size: int = 5):
        """
        Initialize direction stabilizer.
        
        Args:
            history_size: Number of recent directions to consider for stabilization
        """
        self.history_size = history_size
        self.direction_history: Dict[str, deque] = {}
    
    def update_direction(self, object_id: str, detected_direction: Optional[str]):
        """
        Update the direction history for an object.
        
        Args:
            object_id: Unique identifier for the object
            detected_direction: The direction detected in current frame ('left', 'right', or None)
        """
        if object_id not in self.direction_history:
            self.direction_history[object_id] = deque(maxlen=self.history_size)
        
        # Only add non-None directions to history
        if detected_direction is not None:
            self.direction_history[object_id].append(detected_direction)
    
    def get_stable_direction(self, object_id: str) -> Optional[str]:
        """
        Get the most frequent direction from recent history.
        
        Args:
            object_id: Unique identifier for the object
            
        Returns:
            Most frequent direction or None if no directions recorded
        """
        if object_id not in self.direction_history:
            return None
        
        history = list(self.direction_history[object_id])
        if not history:
            return None
        
        # Count occurrences of each direction
        direction_counts = Counter(history)
        
        # Return the most common direction
        most_common = direction_counts.most_common(1)
        return most_common[0][0] if most_common else None
    
    def reset_object(self, object_id: str):
        """
        Reset direction history for a specific object.
        
        Args:
            object_id: Object to reset
        """
        if object_id in self.direction_history:
            del self.direction_history[object_id]
    
    def get_debug_info(self, object_id: str) -> Dict:
        """
        Get debug information for an object.
        
        Args:
            object_id: Object to get debug info for
            
        Returns:
            Dictionary with debug information
        """
        history = list(self.direction_history.get(object_id, []))
        stable_direction = self.get_stable_direction(object_id)
        
        debug_info = {
            'object_id': object_id,
            'direction_history': history,
            'history_length': len(history),
            'stable_direction': stable_direction,
            'history_size': self.history_size
        }
        
        if history:
            direction_counts = Counter(history)
            debug_info['direction_counts'] = dict(direction_counts)
        
        return debug_info


class EnemySubmarineDirectionStabilizer:
    """Specialized direction stabilizer for enemy submarines."""
    
    def __init__(self, history_size: int = 5):
        """
        Initialize enemy submarine direction stabilizer.
        
        Args:
            history_size: Number of recent directions to consider
        """
        self.stabilizer = DirectionStabilizer(history_size)
    
    def update_submarine_direction(self, submarine_object):
        """
        Update direction for an enemy submarine based on its current characteristics.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
        """
        if submarine_object.object_type != 'enemy_submarine':
            return
        
        # Get the direction from visual detection (same as player)
        detected_direction = submarine_object.characteristics.get('facing_side', None)
        
        # Update the stabilizer with this direction
        self.stabilizer.update_direction(submarine_object.object_id, detected_direction)
        
        # Get the stable direction and update the object's characteristics
        stable_direction = self.stabilizer.get_stable_direction(submarine_object.object_id)
        
        if stable_direction:
            # Update the object's characteristics with the stabilized direction
            if not hasattr(submarine_object, 'characteristics'):
                submarine_object.characteristics = {}
            
            submarine_object.characteristics['facing_side'] = stable_direction
            submarine_object.characteristics['facing_source'] = 'visual_stabilized'
    
    def get_submarine_stable_direction(self, submarine_object) -> Optional[str]:
        """
        Get stable facing direction for an enemy submarine.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
            
        Returns:
            Stable direction or None
        """
        if submarine_object.object_type != 'enemy_submarine':
            return None
        
        return self.stabilizer.get_stable_direction(submarine_object.object_id)
    
    def reset_submarine(self, submarine_object):
        """
        Reset direction history for a submarine.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
        """
        if submarine_object.object_type == 'enemy_submarine':
            self.stabilizer.reset_object(submarine_object.object_id)
    
    def get_debug_info_for_submarine(self, submarine_object) -> Dict:
        """
        Get debug information for a submarine.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
            
        Returns:
            Debug information dictionary
        """
        if submarine_object.object_type != 'enemy_submarine':
            return {}
        
        return self.stabilizer.get_debug_info(submarine_object.object_id)
