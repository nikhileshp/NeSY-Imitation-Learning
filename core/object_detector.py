"""
Core object detection module - game-agnostic base classes.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Protocol
from models.OC_Atari.ocatari.vision.utils import find_objects
from .game_object import GameObject


class GameConfig(Protocol):
    """Protocol defining the interface for game-specific configurations."""
    object_colors: Dict[str, List]
    detection_params: Dict[str, Dict[str, Any]]
    
    def get_object_types(self) -> List[str]:
        """Return list of object types for this game."""
        ...


class BaseObjectDetector:
    """Base object detector that can be extended for specific games."""
    
    def __init__(self, game_config: GameConfig):
        """
        Initialize the object detector with game-specific configuration.
        
        Args:
            game_config: Game-specific configuration object
        """
        self.game_config = game_config
    
    def detect_objects_by_type(self, image: np.ndarray, object_type: str) -> List[GameObject]:
        """
        Detect objects of a specific type in the image.
        
        Args:
            image: Input image as numpy array
            object_type: Type of object to detect
            
        Returns:
            List of detected GameObjects
        """
        if object_type not in self.game_config.object_colors:
            return []
        
        colors = self.game_config.object_colors[object_type]
        params = self.game_config.detection_params.get(object_type, {})
        
        coords_list = find_objects(image, colors, **params)
        
        objects = []
        for i, coords in enumerate(coords_list):
            obj = GameObject(object_type, coords, f'{object_type}_{i}')
            objects.append(obj)
        
        return objects
    
    def detect_all_objects(self, image: np.ndarray) -> Dict[str, List[GameObject]]:
        """
        Detect all configured object types in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary mapping object types to lists of detected GameObjects
        """
        detected_objects = {}
        
        for object_type in self.game_config.get_object_types():
            detected_objects[object_type] = self.detect_objects_by_type(image, object_type)
        
        return detected_objects
    
    def get_all_objects_as_list(self, detected_objects: Dict[str, List[GameObject]]) -> List[str]:
        """
        Get a flat list of all object IDs for easier processing.
        
        Args:
            detected_objects: Dictionary of detected objects
            
        Returns:
            List of all object IDs
        """
        all_objects = []
        for object_type, objects in detected_objects.items():
            for obj in objects:
                all_objects.append(obj.object_id)
        
        return all_objects
    
    def filter_overlapping_objects(self, objects1: List[GameObject], 
                                  objects2: List[GameObject]) -> List[GameObject]:
        """
        Remove objects from objects1 that overlap with any object in objects2.
        
        Args:
            objects1: List of objects to filter
            objects2: List of objects to check overlap against
            
        Returns:
            Filtered list of objects from objects1
        """
        filtered_objects = []
        
        for obj1 in objects1:
            overlaps = False
            for obj2 in objects2:
                if self._objects_overlap(obj1, obj2):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_objects.append(obj1)
        
        return filtered_objects
    
    def _objects_overlap(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Check if two game objects overlap."""
        return (obj1.left < obj2.right and obj2.left < obj1.right and
                obj1.top < obj2.bottom and obj2.top < obj1.bottom)
