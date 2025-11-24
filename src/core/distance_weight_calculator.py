"""
Distance weight calculator for computing gaze-object relationship weights.
"""
import math
from typing import List, Tuple, Dict, Optional
from .game_object import GameObject, SpatialRelationship


class DistanceWeightCalculator:
    """
    Calculates distance weights for relationships between gaze coordinates and spatial objects.
    Weight formula: max_possible_distance / actual_distance
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the distance weight calculator.
        
        Args:
            screen_width: Width of the game screen in pixels
            screen_height: Height of the game screen in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_possible_distance = math.sqrt(screen_width ** 2 + screen_height ** 2)
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance between the points
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def calculate_distance_weight(self, gaze_pos: Tuple[int, int], obj_center: Tuple[int, int]) -> float:
        """
        Calculate distance weight between gaze position and object center.
        
        Args:
            gaze_pos: Gaze position (x, y)
            obj_center: Object center position (x, y)
            
        Returns:
            Distance weight (max_possible_distance / actual_distance)
        """
        actual_distance = self.calculate_distance(gaze_pos, obj_center)
        
        # Handle case where gaze is exactly on object center
        if actual_distance == 0:
            return self.max_possible_distance
        
        return self.max_possible_distance / actual_distance
    
    def calculate_reciprocal_rank_weights(self, gaze_pos: Tuple[int, int], 
                                        objects_with_centers: List[Tuple[GameObject, Tuple[int, int]]]) -> Dict[GameObject, float]:
        """
        Calculate reciprocal rank-based distance weights for objects.
        Closest object gets weight 1, second closest gets 1/2, third gets 1/3, etc.
        
        Args:
            gaze_pos: Gaze position (x, y)
            objects_with_centers: List of (GameObject, center_position) tuples
            
        Returns:
            Dictionary mapping GameObjects to their reciprocal rank weights
        """
        if not objects_with_centers:
            return {}
        
        # Calculate distances and sort by distance
        object_distances = []
        for obj, center in objects_with_centers:
            distance = self.calculate_distance(gaze_pos, center)
            object_distances.append((obj, center, distance))
        
        # Sort by distance (ascending - closest first)
        object_distances.sort(key=lambda x: x[2])
        
        # Assign reciprocal rank weights
        weights = {}
        for rank, (obj, center, distance) in enumerate(object_distances, 1):
            weights[obj] = 1.0 / rank
        
        return weights
    
    def calculate_alternating_class_weights(self, gaze_pos: Tuple[int, int], 
                                          objects_with_centers: List[Tuple[GameObject, Tuple[int, int]]]) -> Dict[GameObject, float]:
        """
        Calculate alternating class-based reciprocal rank weights for objects.
        First object of each class gets weight, second object of same class gets 0, 
        third object of same class gets next available weight, etc.
        
        Args:
            gaze_pos: Gaze position (x, y)
            objects_with_centers: List of (GameObject, center_position) tuples
            
        Returns:
            Dictionary mapping GameObjects to their alternating class weights
        """
        if not objects_with_centers:
            return {}
        
        # Calculate distances and sort by distance
        object_distances = []
        for obj, center in objects_with_centers:
            distance = self.calculate_distance(gaze_pos, center)
            object_distances.append((obj, center, distance))
        
        # Sort by distance (ascending - closest first)
        object_distances.sort(key=lambda x: x[2])
        
        # Track class occurrence counts and assign weights
        weights = {}
        class_counts = {}  # Track how many objects of each class we've seen
        current_rank = 1   # Current reciprocal rank to assign
        
        for obj, center, distance in object_distances:
            object_class = obj.object_type
            
            # Initialize class count if not seen before
            if object_class not in class_counts:
                class_counts[object_class] = 0
            
            class_counts[object_class] += 1
            class_occurrence = class_counts[object_class]
            
            # Assign weight based on class occurrence pattern
            if class_occurrence % 2 == 1:  # Odd occurrence (1st, 3rd, 5th, etc.)
                weights[obj] = 1.0 / current_rank
                current_rank += 1
            else:  # Even occurrence (2nd, 4th, 6th, etc.)
                weights[obj] = 0.0
        
        return weights
    
    def calculate_nearest_only_weights(self, gaze_pos: Tuple[int, int], 
                                     objects_with_centers: List[Tuple[GameObject, Tuple[int, int]]]) -> Dict[GameObject, float]:
        """
        Calculate nearest-only distance weights for objects.
        Only the nearest object gets weight 1.0, all others get weight 0.0.
        
        Args:
            gaze_pos: Gaze position (x, y)
            objects_with_centers: List of (GameObject, center_position) tuples
            
        Returns:
            Dictionary mapping GameObjects to their nearest-only weights
        """
        if not objects_with_centers:
            return {}
        
        # Calculate distances and find the nearest object
        object_distances = []
        for obj, center in objects_with_centers:
            distance = self.calculate_distance(gaze_pos, center)
            object_distances.append((obj, center, distance))
        
        # Sort by distance (ascending - closest first)
        object_distances.sort(key=lambda x: x[2])
        
        # Assign weights: only the nearest gets 1.0, all others get 0.0
        weights = {}
        for i, (obj, center, distance) in enumerate(object_distances):
            if i == 0:  # Nearest object
                weights[obj] = 1.0
            else:  # All other objects
                weights[obj] = 0.0
        
        return weights
    def calculate_relationship_distance_weights(self, relationships: List[SpatialRelationship], 
                                               gaze_positions: List[Tuple[int, int]], 
                                               use_alternating_class_weights: bool = False,
                                               use_nearest_only_weights: bool = False) -> Dict[str, float]:
        """
        Calculate distance weights for relationships involving spatial objects.
        Uses the last gaze position (the one being displayed) for calculation.
        
        Args:
            relationships: List of spatial relationships
            gaze_positions: List of gaze positions for the frame
            use_alternating_class_weights: If True, use alternating class weight calculation
                                         (first object of each class gets weight, second gets 0, etc.)
            use_nearest_only_weights: If True, only the nearest object gets weight 1, all others get 0
            
        Returns:
            Dictionary mapping individual relationship identifiers to distance weights
        """
        if not gaze_positions:
            return {}
        
        # Use the last gaze position (the one being displayed)
        gaze_pos = gaze_positions[-1]
        
        # Define which object types we want to calculate distance weights for
        target_object_types = {'diver', 'enemy', 'enemy_submarine', 'enemy_missile'}
        
        # Collect all spatial objects from relationships with their centers
        spatial_objects_with_centers = []
        relationship_to_object_map = {}  # Maps (rel_type, obj_id) to the relationship
        
        for relationship in relationships:
            obj1 = relationship.obj1
            obj2 = relationship.obj2
            rel_type = relationship.relationship_type
            
            # Check if this relationship involves spatial objects we care about
            spatial_object = None
            
            # Check if obj1 is a spatial object we care about
            if obj1.object_type in target_object_types:
                spatial_object = obj1
            # Check if obj2 is a spatial object (for visibility relationships)
            elif obj2.object_type in target_object_types:
                spatial_object = obj2
            
            # Skip relationships that don't involve our target object types
            if spatial_object is None:
                continue
            
            # Add to our collection if not already present
            obj_center = spatial_object.center
            if spatial_object not in [obj for obj, _ in spatial_objects_with_centers]:
                spatial_objects_with_centers.append((spatial_object, obj_center))
            
            # Map relationship identifier to relationship for later lookup
            rel_key = (rel_type, spatial_object.object_id)
            relationship_to_object_map[rel_key] = spatial_object
        
        # Calculate weights for all spatial objects based on the selected method
        if use_nearest_only_weights:
            object_weights = self.calculate_nearest_only_weights(gaze_pos, spatial_objects_with_centers)
        elif use_alternating_class_weights:
            object_weights = self.calculate_alternating_class_weights(gaze_pos, spatial_objects_with_centers)
        else:
            # Default: use reciprocal rank weights
            object_weights = self.calculate_reciprocal_rank_weights(gaze_pos, spatial_objects_with_centers)
        
        # Map weights back to relationship identifiers
        distance_weights = {}
        for relationship in relationships:
            obj1 = relationship.obj1
            obj2 = relationship.obj2
            rel_type = relationship.relationship_type
            
            # Find the spatial object for this relationship
            spatial_object = None
            if obj1.object_type in target_object_types:
                spatial_object = obj1
            elif obj2.object_type in target_object_types:
                spatial_object = obj2
            
            if spatial_object is not None and spatial_object in object_weights:
                # Create unique identifier for this relationship instance
                rel_identifier = f"{rel_type}({spatial_object.object_id})"
                distance_weights[rel_identifier] = object_weights[spatial_object]
        
        return distance_weights
    
    def format_distance_weights_for_dataframe(self, distance_weights: Dict[str, float]) -> str:
        """
        Format distance weights for storage in a DataFrame.
        
        Args:
            distance_weights: Dictionary of relationship identifiers to weights
            
        Returns:
            Formatted string of distance weights
        """
        if not distance_weights:
            return ""
        
        # Format as rel_identifier:weight pairs
        formatted_parts = []
        for rel_identifier, weight in distance_weights.items():
            formatted_parts.append(f"{rel_identifier} {weight:.2f}")
        
        return " , ".join(formatted_parts)
