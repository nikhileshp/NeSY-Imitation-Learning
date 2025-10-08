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
    
    def calculate_relationship_distance_weights(self, relationships: List[SpatialRelationship], 
                                               gaze_positions: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Calculate distance weights for relationships involving spatial objects.
        Uses the last gaze position (the one being displayed) for calculation.
        
        Args:
            relationships: List of spatial relationships
            gaze_positions: List of gaze positions for the frame
            
        Returns:
            Dictionary mapping individual relationship identifiers to distance weights
        """
        if not gaze_positions:
            return {}
        
        # Use the last gaze position (the one being displayed)
        gaze_pos = gaze_positions[-1]
        
        # Define which object types we want to calculate distance weights for
        target_object_types = {'diver', 'enemy', 'enemy_submarine', 'enemy_missile'}
        
        # Dictionary to store distance weights by individual relationship
        distance_weights = {}
        
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
            
            # Calculate distance weight from the displayed gaze position to this object
            obj_center = spatial_object.center
            weight = self.calculate_distance_weight(gaze_pos, obj_center)
            
            # Create unique identifier for this relationship instance
            # Include object ID to distinguish between multiple objects of same type
            rel_identifier = f"{rel_type}({spatial_object.object_id})"
            distance_weights[rel_identifier] = weight
        
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
            formatted_parts.append(f"{rel_identifier}:{weight:.2f}")
        
        return " ; ".join(formatted_parts)
