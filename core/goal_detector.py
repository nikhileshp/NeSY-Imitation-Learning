"""
Goal detection module for inferring player goals based on gaze data and game state.
"""
import math
from typing import List, Dict, Tuple, Optional, Any
from core.game_object import GameObject
from env.seaquest.config import WATER_SURFACE_Y


class GoalDetector:
    """Detects player goals based on gaze data and game state analysis."""
    
    def __init__(self):
        """Initialize the goal detector."""
        self.previous_gaze_positions = []
    
    def detect_goal(self, gaze_positions: List[Tuple[int, int]], 
                   detected_objects: Dict[str, List[GameObject]],
                   action: int,
                   frame_id: str) -> str:
        """
        Detect player goal based on gaze data and game state.
        
        Args:
            gaze_positions: List of (x, y) gaze position tuples for current frame
            detected_objects: Dictionary mapping object types to lists of GameObjects
            action: Player action (for determining if shooting)
            frame_id: Current frame identifier
            
        Returns:
            String representing the detected goal
        """
        if not gaze_positions:
            self.previous_gaze_positions = []
            return "unknown"
        
        # Find the farthest gaze point from previous frame
        current_gaze_point = self._get_farthest_gaze_point(gaze_positions)
        
        # Update previous gaze positions for next frame
        self.previous_gaze_positions = gaze_positions.copy()
        
        if not current_gaze_point:
            return "unknown"
        
        # Find the closest object to the gaze point (skipping player if it's closest)
        closest_object = self._find_closest_object(current_gaze_point, detected_objects)
        
        # Determine goal based on object and game state
        goal = self._determine_goal(closest_object, detected_objects, action, current_gaze_point)
        
        # print(f"Frame {frame_id}: Gaze at {current_gaze_point}, closest object: "
            #   f"{closest_object.object_type if closest_object else 'None'}, goal: {goal}")
        
        return goal
    
    def _get_farthest_gaze_point(self, gaze_positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Find the gaze point that is farthest from all previous frame gaze points.
        
        Args:
            gaze_positions: Current frame gaze positions
            
        Returns:
            (x, y) tuple of the farthest gaze point, or None if no valid point
        """
        if not gaze_positions:
            return None
        
        # If no previous gaze positions, return the first current position
        if not self.previous_gaze_positions:
            return gaze_positions[0]
        
        max_min_distance = -1
        farthest_point = None
        
        # For each current gaze point, find its minimum distance to any previous point
        for current_point in gaze_positions:
            min_distance_to_previous = min(
                self._calculate_distance(current_point, prev_point) 
                for prev_point in self.previous_gaze_positions
            )
            
            # Select the point with the maximum minimum distance (farthest from all previous points)
            if min_distance_to_previous > max_min_distance:
                max_min_distance = min_distance_to_previous
                farthest_point = current_point
        
        return farthest_point
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _find_closest_object(self, gaze_point: Tuple[int, int], 
                            detected_objects: Dict[str, List[GameObject]]) -> Optional[GameObject]:
        """
        Find the closest detected object to the gaze point.
        If the closest object is a player, return the second closest object instead.
        
        Args:
            gaze_point: (x, y) gaze position
            detected_objects: Dictionary mapping object types to lists of GameObjects
            
        Returns:
            Closest non-player GameObject, or second closest if closest is player
        """
        # Create list of all objects with their distances
        object_distances = []
        
        # Check all detected objects
        for object_type, objects in detected_objects.items():
            for obj in objects:
                # Calculate distance from gaze point to object center
                obj_center = self._get_object_center(obj)
                distance = self._calculate_distance(gaze_point, obj_center)
                object_distances.append((distance, obj))
        
        # Sort by distance (closest first)
        object_distances.sort(key=lambda x: x[0])
        
        if not object_distances:
            return None
        
        # If closest object is not a player, return it
        closest_obj = object_distances[0][1]
        if closest_obj.object_type != 'player':
            return closest_obj
        
        # If closest is player, return second closest (if available)
        if len(object_distances) > 1:
            second_closest_obj = object_distances[1][1]
            return second_closest_obj
        
        # Only player objects detected, return None to indicate no meaningful object
        return None
    
    def _get_object_center(self, obj: GameObject) -> Tuple[int, int]:
        """Get the center point of a game object's bounding box."""
        x, y, w, h = obj.bounding_box
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        return (center_x, center_y)
    
    def _determine_goal(self, closest_object: Optional[GameObject], 
                       detected_objects: Dict[str, List[GameObject]], 
                       action: int, gaze_point: Tuple[int, int]) -> str:
        """
        Determine the player's goal based on what they're looking at and game state.
        
        Args:
            closest_object: The object closest to the gaze point
            detected_objects: All detected objects in the current frame
            action: Player action (for determining if shooting)
            gaze_point: Current gaze position
            
        Returns:
            String representing the inferred goal
        """
        # Get game state information
        divers = detected_objects.get('diver', [])
        collected_divers = detected_objects.get('collected_diver', [])
        enemies = detected_objects.get('enemy', []) + detected_objects.get('enemy_submarine', [])
        players = detected_objects.get('player', [])
        
        # Check diver states
        divers_full = len(collected_divers) >= 6
        divers_empty = len(collected_divers) == 0
        
        # Check if looking above water surface
        looking_above_water = gaze_point[1] < WATER_SURFACE_Y
        
        # Check if player/submarine is above water (for waitForOxygen detection)
        player_above_water = False
        if players:
            player_y = players[0].bounding_box[1]  # y coordinate of player
            player_above_water = player_y < WATER_SURFACE_Y
        
        # Check if player is shooting (action codes for shooting vary by game)
        # In Seaquest, action 1 is typically FIRE
        is_shooting = action == 1
        
        # Priority Rule 1: If divers are full, goal is always surface
        if divers_full:
            return "surface"
        
        # Priority Rule 2: If divers are empty and player is above water, goal is waitForOxygen
        if divers_empty and player_above_water:
            return "waitForOxygen"
        
        if not closest_object:
            # No object detected near gaze point
            if looking_above_water and divers_empty:
                return "waitForOxygen"
            return "unknown"
        
        object_type = closest_object.object_type
        
        # Goal detection rules based on requirements:
        
        # 1. If looking at a diver, goal is retrieve_diver
        if object_type == 'diver':
            return "retrieve_diver"
        
        # 2. If looking at an enemy when there's no visible diver, goal is kill_enemy
        if object_type in ['enemy', 'enemy_submarine']:
            if not divers:  # No visible divers
                return "kill_enemy"
            else:
                # There are visible divers
                if not is_shooting:
                    return "avoid_enemy"
                else:
                    # Looking at enemy and shooting despite visible divers - could still be kill_enemy
                    return "kill_enemy"
        
        # 3. If looking above water when divers are empty, goal is waitForOxygen
        if looking_above_water and divers_empty:
            return "waitForOxygen"
        
        # 4. Default case - if looking at other objects or unclear situation
        if object_type in ['player_missile', 'enemy_missile']:
            # Looking at missiles - could be avoiding or tracking threats
            if enemies:
                return "avoid_enemy"
            return "unknown"
        
        # Fallback for any other object types or unclear situations
        return "unknown"
