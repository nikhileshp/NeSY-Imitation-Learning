"""
Seaquest-specific object detector implementation.
"""
import numpy as np
from typing import List, Dict
from models.ocatari.ocatari.vision.utils import find_objects, facing_side

from core.object_detector import BaseObjectDetector, GameConfig
from core.game_object import GameObject
from core.object_tracker import ObjectTracker
from .config import OBJECT_COLORS, ENEMY_COLORS, DETECTION_PARAMS


class SeaquestGameConfig:
    """Game configuration for Seaquest."""
    
    def __init__(self):
        self.object_colors = OBJECT_COLORS
        self.enemy_colors = ENEMY_COLORS
        self.detection_params = DETECTION_PARAMS
    
    def get_object_types(self) -> List[str]:
        """Return list of object types for Seaquest."""
        return [
            'player', 'diver', 'collected_diver', 'player_missile', 
            'enemy_missile', 'lives', 'enemy_submarine', 'oxygen_bar', 
            'oxygen_depleted', 'enemy'
        ]


class SeaquestObjectDetector(BaseObjectDetector):
    """Seaquest-specific object detector with custom detection logic and object tracking."""
    
    def __init__(self):
        """Initialize with Seaquest configuration."""
        super().__init__(SeaquestGameConfig())
        
        # Initialize object tracker with maximum object counts for each type
        max_objects = {
            'player': 1,
            'enemy': 20,          # Allow up to 20 enemies
            'enemy_submarine': 10, # Allow up to 10 submarines
            'diver': 15,           # Allow up to 15 divers
            'collected_diver': 8,  # Max 8 collected divers (game limit is 6)
            'player_missile': 5,   # Allow up to 5 player missiles
            'enemy_missile': 20,   # Allow up to 20 enemy missiles
            'lives': 5,            # Max 5 lives display
            'oxygen_bar': 1,
            'oxygen_depleted': 1
        }
        self.object_tracker = ObjectTracker(max_objects)
        self.use_tracking = True
    
    def detect_objects_by_type(self, image, object_type):
        """
        Detect objects of a specific type in the image.
        
        Args:
            image: Input image as numpy array
            object_type: Type of object to detect
            
        Returns:git 
            List of detected GameObjects
        """
        if object_type not in self.game_config.object_colors:
            return []
        
        colors = self.game_config.object_colors[object_type]
        params = self.game_config.detection_params.get(object_type, {})
        coords_list = find_objects(image, colors, **params)

        side = facing_side(image, colors, coords_list)

        objects = []
        for i, coords in enumerate(coords_list):
            if object_type in ['player']:


                obj = GameObject(object_type=object_type, bounding_box=coords, object_id=f'{object_type}_{i}', characteristics={'facing_side': side})
            else:
                # print(object_type)
                obj = GameObject(object_type=object_type, bounding_box=coords, object_id=f'{object_type}_{i}')
            objects.append(obj)
            # print(objects)

        return objects
        
    
    def detect_all_objects(self, image: np.ndarray) -> Dict[str, List[GameObject]]:
        
        """
        Detect all Seaquest objects with custom logic.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary mapping object types to lists of detected GameObjects
        """

        detected_objects = {}
        
        # Detect basic objects using base detector
        detected_objects['player'] = self.detect_objects_by_type(image, 'player')
        detected_objects['diver'] = self.detect_objects_by_type(image, 'diver')
        detected_objects['collected_diver'] = self.detect_objects_by_type(image, 'collected_diver')
        detected_objects['player_missile'] = self.detect_objects_by_type(image, 'player_missile')
        detected_objects['lives'] = self.detect_objects_by_type(image, 'lives')
        detected_objects['oxygen_bar'] = self.detect_objects_by_type(image, 'oxygen_bar')
        detected_objects['oxygen_depleted'] = self.detect_objects_by_type(image, 'oxygen_depleted')
        
        # Detect submarines with special logic
        detected_objects['enemy_submarine'] = self._detect_submarines(image)
        
        # Detect enemy missiles with special filtering logic
        detected_objects['enemy_missile'] = self._detect_enemy_missiles(image, detected_objects['diver'])
        
        # Detect enemies using combined colors
        detected_objects['enemy'] = self._detect_enemies(image)
        
        # Apply Seaquest-specific cleanup
        self._cleanup_detections(detected_objects)
        
        # Apply object tracking to maintain consistent IDs across frames
        if self.use_tracking:
            detected_objects = self.object_tracker.track_all_objects(detected_objects)
       
        return detected_objects
    
    def _detect_submarines(self, image: np.ndarray) -> List[GameObject]:
        """Detect submarine objects with underwater and surface detection."""
        submarines = []
        all_submarine_coords = []
        
        # Detect underwater submarines
        underwater_params = self.game_config.detection_params.get('submarine', {})
        underwater_coords = find_objects(image, self.game_config.object_colors['submarine'], 
                                       **underwater_params)
        all_submarine_coords.extend(underwater_coords)
        

        # Detect submarines on water surface
        surface_params = self.game_config.detection_params.get('submarine_on_water', {})
        surface_coords = find_objects(image, self.game_config.object_colors['submarine'], 
                                    **surface_params)
        all_submarine_coords.extend(surface_coords)
        
        # Create GameObjects with sequential numbering
        # Note: We don't set facing_side here to allow the ObjectTracker's 
        # SimpleSubmarineDirectionDetector to determine direction based on x-coordinate
        for i, coords in enumerate(all_submarine_coords):
            submarine = GameObject(object_type='enemy_submarine', bounding_box=coords, object_id=f'enemy_submarine_{i}')
            submarines.append(submarine)
        
        return submarines
    
    def _detect_enemy_missiles(self, image: np.ndarray, divers: List[GameObject]) -> List[GameObject]:
        """Detect enemy missiles with special filtering logic."""
        params = self.game_config.detection_params.get('enemy_missile', {})
        coords_list = find_objects(image, self.game_config.object_colors['enemy_missile'], **params)
        
        # First, collect all valid missile coordinates
        valid_missile_coords = []
        
        # Add missile coordinates that don't overlap with divers
        for coords in coords_list:
            temp_missile = GameObject('enemy_missile', coords, 'temp')
            overlaps_with_diver = False
            
            for diver in divers:
                if self._objects_overlap(temp_missile, diver):
                    overlaps_with_diver = True
                    break
            
            if not overlaps_with_diver:
                valid_missile_coords.append(coords)
        
        # Add small divers that are actually enemy missiles (Seaquest-specific logic)
        if divers:
            for diver in divers:
                if (6 <= diver.width <= 8) and diver.height == 4:
                    valid_missile_coords.append(diver.bounding_box)
        
        # Create GameObjects with sequential numbering
        enemy_missiles = []
        for i, coords in enumerate(valid_missile_coords):
            missile = GameObject('enemy_missile', coords, f'enemy_missile_{i}')
            enemy_missiles.append(missile)
        
        return enemy_missiles
    
    def _detect_enemies(self, image: np.ndarray) -> List[GameObject]:
        """Detect enemy objects using combined enemy colors."""
        params = self.game_config.detection_params.get('enemy', {})
        enemies = []
        
        # Detect different types of enemies by color
        enemy_coords = []
        enemy_coords.extend(find_objects(image, self.game_config.enemy_colors['green'], **params))
        enemy_coords.extend(find_objects(image, self.game_config.enemy_colors['lightgreen'], **params))
        enemy_coords.extend(find_objects(image, self.game_config.enemy_colors['pink'], **params))
        
        # Combine orange and yellow colors
        orange_yellow = self.game_config.enemy_colors['orange'] + self.game_config.enemy_colors['yellow']
        enemy_coords.extend(find_objects(image, orange_yellow, **params))
        
        for i, coords in enumerate(enemy_coords):
            enemy = GameObject('enemy', coords, f'enemy_{i}')
            enemies.append(enemy)
        
        return enemies
    
    def _cleanup_detections(self, detected_objects: Dict[str, List[GameObject]]):
        """Apply Seaquest-specific cleanup logic."""
        # Remove player missiles that overlap with player
        if detected_objects.get('player') and detected_objects.get('player_missile'):
            cleaned_missiles = self.filter_overlapping_objects(
                detected_objects['player_missile'],
                detected_objects['player']
            )
            # Reassign sequential IDs after filtering
            for i, missile in enumerate(cleaned_missiles):
                missile.object_id = f'player_missile_{i}'
            detected_objects['player_missile'] = cleaned_missiles
        
        # Reassign sequential IDs for all object types to ensure consistency
        for object_type, objects in detected_objects.items():
            for i, obj in enumerate(objects):
                obj.object_id = f'{object_type}_{i}'
    
    def _objects_overlap(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Check if two game objects overlap (Seaquest-specific logic)."""
        return (obj1.left >= obj2.left and obj1.left <= obj2.right and 
                obj1.top >= obj2.top and obj1.top <= obj2.bottom)
    
    def enable_tracking(self):
        """Enable object tracking to maintain consistent IDs."""
        self.use_tracking = True
    
    def disable_tracking(self):
        """Disable object tracking (use sequential IDs)."""
        self.use_tracking = False
    
    def reset_tracking(self):
        """Reset the object tracker state."""
        self.object_tracker.reset()
    
    def get_tracking_info(self):
        """Get current tracking information for debugging."""
        return self.object_tracker.get_tracking_info()
