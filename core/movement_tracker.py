"""
Movement-based facing detection system for enemy submarines.
Tracks bounding box positions across frames to determine facing direction.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque


class MovementTracker:
    """Tracks object movement to determine facing direction."""
    
    def __init__(self, history_size: int = 5, min_movement_threshold: float = 2.0, momentum_threshold: int = 3, lock_direction: bool = False):
        """
        Initialize movement tracker.
        
        Args:
            history_size: Number of previous positions to track
            min_movement_threshold: Minimum movement distance to consider for direction
            momentum_threshold: Number of consistent direction changes needed to update facing
            lock_direction: If True, direction is locked once detected and never changes
        """
        self.history_size = history_size
        self.min_movement_threshold = min_movement_threshold
        self.momentum_threshold = momentum_threshold
        self.lock_direction = lock_direction
        self.position_history: Dict[str, deque] = {}
        self.facing_direction_cache: Dict[str, str] = {}
        self.direction_momentum: Dict[str, deque] = {}  # Track direction changes for momentum
        self.direction_locked: Dict[str, bool] = {}  # Track which objects have locked directions
    
    def update_position(self, object_id: str, center_x: float, center_y: float):
        """
        Update position history for an object.
        
        Args:
            object_id: Unique identifier for the object
            center_x: X coordinate of object center
            center_y: Y coordinate of object center
        """
        if object_id not in self.position_history:
            self.position_history[object_id] = deque(maxlen=self.history_size)
        
        self.position_history[object_id].append((center_x, center_y))
    
    def get_facing_direction(self, object_id: str) -> Optional[str]:
        """
        Determine facing direction based on X-coordinate movement only.
        
        If lock_direction is True, the direction is determined once and never changes.
        Otherwise, uses momentum-based stability to avoid flickering when submarines briefly move backward.
        
        Args:
            object_id: Unique identifier for the object
            
        Returns:
            Facing direction ('left' or 'right') or None if insufficient data
        """
        if object_id not in self.position_history:
            return None
        
        # If direction locking is enabled and this object already has a locked direction, return it
        if self.lock_direction and self.direction_locked.get(object_id, False):
            return self.facing_direction_cache.get(object_id)
        
        history = list(self.position_history[object_id])
        
        # Need at least 2 positions to determine movement
        if len(history) < 2:
            return self.facing_direction_cache.get(object_id, None)
        
        # Calculate X-coordinate changes only
        x_movements = []
        for i in range(1, len(history)):
            prev_x, prev_y = history[i-1]
            curr_x, curr_y = history[i]
            
            dx = curr_x - prev_x
            
            # Only consider significant X movements
            if abs(dx) >= self.min_movement_threshold:
                x_movements.append(dx)
        
        if not x_movements:
            # No significant movement, return cached direction
            return self.facing_direction_cache.get(object_id, None)
        
        # Determine direction based on movement patterns
        if self.lock_direction:
            # For locked direction mode, analyze first 5 frames of movement to find max direction
            if len(history) >= 5:  # Wait for 5 frames before determining direction
                # Calculate cumulative movement in each direction over first 5 frames
                left_movement = 0
                right_movement = 0
                
                for i in range(1, min(6, len(history))):
                    prev_x, prev_y = history[i-1]
                    curr_x, curr_y = history[i]
                    dx = curr_x - prev_x
                    
                    # Only consider significant movements
                    if abs(dx) >= self.min_movement_threshold:
                        if dx < 0:
                            left_movement += abs(dx)
                        else:
                            right_movement += abs(dx)
                
                # Determine direction based on maximum movement
                if left_movement > right_movement:
                    direction = 'left'
                elif right_movement > left_movement:
                    direction = 'right'
                else:
                    # If movements are equal, fall back to average of all movements
                    avg_dx = np.mean(x_movements) if x_movements else 0
                    direction = 'left' if avg_dx < 0 else 'right'
                
                # Lock the direction now that we have 5 frames of data
                self.facing_direction_cache[object_id] = direction
                self.direction_locked[object_id] = True
                return direction
            else:
                # Not enough frames yet, return cached direction if available, otherwise None
                return self.facing_direction_cache.get(object_id, None)
        else:
            # Use momentum-based approach for dynamic direction changes
            recent_avg_dx = np.mean(x_movements[-2:]) if len(x_movements) >= 2 else x_movements[-1]
            immediate_direction = 'left' if recent_avg_dx < 0 else 'right'
            
            # Initialize momentum tracking for this object if not exists
            if object_id not in self.direction_momentum:
                self.direction_momentum[object_id] = deque(maxlen=self.momentum_threshold * 2)
            
            # Add the immediate direction to momentum tracking
            self.direction_momentum[object_id].append(immediate_direction)
            
            # Current cached direction
            current_cached_direction = self.facing_direction_cache.get(object_id)
            
            # If we don't have a cached direction, use immediate direction
            if current_cached_direction is None:
                self.facing_direction_cache[object_id] = immediate_direction
                return immediate_direction
            
            # Count consecutive occurrences of the immediate direction
            momentum_history = list(self.direction_momentum[object_id])
            if len(momentum_history) < self.momentum_threshold:
                # Not enough momentum history, keep current direction
                return current_cached_direction
            
            # Check if the last N directions are consistent and different from cached
            recent_directions = momentum_history[-self.momentum_threshold:]
            if (len(set(recent_directions)) == 1 and  # All same direction
                recent_directions[0] != current_cached_direction and  # Different from cached
                all(d == immediate_direction for d in recent_directions)):  # Consistent with immediate
                # Enough momentum to change direction
                self.facing_direction_cache[object_id] = immediate_direction
                return immediate_direction
            
            # Not enough momentum to change, keep cached direction
            return current_cached_direction
    
    
    def reset_object(self, object_id: str):
        """
        Reset tracking data for a specific object.
        
        Args:
            object_id: Object to reset
        """
        if object_id in self.position_history:
            del self.position_history[object_id]
        if object_id in self.facing_direction_cache:
            del self.facing_direction_cache[object_id]
        if object_id in self.direction_momentum:
            del self.direction_momentum[object_id]
        if object_id in self.direction_locked:
            del self.direction_locked[object_id]
    
    def get_debug_info(self, object_id: str) -> Dict:
        """
        Get debug information for an object.
        
        Args:
            object_id: Object to get debug info for
            
        Returns:
            Dictionary with debug information
        """
        history = list(self.position_history.get(object_id, []))
        cached_direction = self.facing_direction_cache.get(object_id, None)
        momentum_history = list(self.direction_momentum.get(object_id, []))
        is_locked = self.direction_locked.get(object_id, False)
        
        debug_info = {
            'object_id': object_id,
            'position_history': history,
            'cached_direction': cached_direction,
            'history_length': len(history),
            'momentum_history': momentum_history,
            'momentum_threshold': self.momentum_threshold,
            'lock_direction': self.lock_direction,
            'direction_locked': is_locked
        }
        
        if len(history) >= 2:
            # Calculate X-coordinate movement only
            prev_x, prev_y = history[-2]
            curr_x, curr_y = history[-1]
            dx = curr_x - prev_x
            
            debug_info.update({
                'last_x_movement': dx,
                'x_movement_significant': abs(dx) >= self.min_movement_threshold
            })
        
        return debug_info


class EnemySubmarineFacingDetector:
    """Specialized facing detection for enemy submarines using movement tracking."""
    
    def __init__(self, lock_direction: bool = True):
        """Initialize the enemy submarine facing detector.
        
        Args:
            lock_direction: If True, direction is locked once detected (recommended for submarines)
        """
        self.movement_tracker = MovementTracker(
            history_size=6,  # Need at least 6 positions to analyze 5 frame transitions
            min_movement_threshold=1.0,  # Lower threshold for submarine movement
            momentum_threshold=3,  # Require 3 consistent direction changes to switch (if not locked)
            lock_direction=lock_direction  # Lock direction once determined
        )
    
    def update_submarine_position(self, submarine_object):
        """
        Update position for an enemy submarine object.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
        """
        if submarine_object.object_type == 'enemy_submarine':
            center_x, center_y = submarine_object.center
            self.movement_tracker.update_position(submarine_object.object_id, center_x, center_y)
    
    def get_submarine_facing_direction(self, submarine_object) -> Optional[str]:
        """
        Get facing direction for an enemy submarine.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
            
        Returns:
            Facing direction or None
        """
        if submarine_object.object_type != 'enemy_submarine':
            return None
        
        return self.movement_tracker.get_facing_direction(submarine_object.object_id)
    
    def update_submarine_characteristics(self, submarine_object):
        """
        Update submarine object characteristics with movement-based facing direction.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
        """
        facing_direction = self.get_submarine_facing_direction(submarine_object)
        
        if facing_direction:
            # Update the object's characteristics with the movement-based facing direction
            if not hasattr(submarine_object, 'characteristics'):
                submarine_object.characteristics = {}
            
            submarine_object.characteristics['facing_side'] = facing_direction
            submarine_object.characteristics['facing_source'] = 'movement_tracking'
    
    def get_debug_info_for_submarine(self, submarine_object) -> Dict:
        """
        Get debug information for a submarine.
        
        Args:
            submarine_object: GameObject representing the enemy submarine
            
        Returns:
            Debug information dictionary
        """
        return self.movement_tracker.get_debug_info(submarine_object.object_id)
