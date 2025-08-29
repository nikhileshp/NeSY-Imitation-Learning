"""
Seaquest-specific object detector implementation.
"""
import numpy as np
from typing import List, Dict
from models.OC_Atari.ocatari.vision.utils import find_objects

from core.object_detector import BaseObjectDetector, GameConfig
from core.game_object import GameObject
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
    """Seaquest-specific object detector with custom detection logic."""
    
    def __init__(self):
        """Initialize with Seaquest configuration."""
        super().__init__(SeaquestGameConfig())
    
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
        
        return detected_objects
    
    def _detect_submarines(self, image: np.ndarray) -> List[GameObject]:
        """Detect submarine objects with underwater and surface detection."""
        submarines = []
        
        # Detect underwater submarines
        underwater_params = self.game_config.detection_params.get('submarine', {})
        underwater_coords = find_objects(image, self.game_config.object_colors['submarine'], 
                                       **underwater_params)
        
        for i, coords in enumerate(underwater_coords):
            submarine = GameObject('enemy_submarine', coords, f'enemy_submarine_{i}')
            submarines.append(submarine)
        
        # Detect submarines on water surface
        surface_params = self.game_config.detection_params.get('submarine_on_water', {})
        surface_coords = find_objects(image, self.game_config.object_colors['submarine'], 
                                    **surface_params)
        
        for coords in surface_coords:
            submarine = GameObject('enemy_submarine', coords, 
                                 f'enemy_submarine_{len(submarines)}')
            submarines.append(submarine)
        
        return submarines
    
    def _detect_enemy_missiles(self, image: np.ndarray, divers: List[GameObject]) -> List[GameObject]:
        """Detect enemy missiles with special filtering logic."""
        params = self.game_config.detection_params.get('enemy_missile', {})
        coords_list = find_objects(image, self.game_config.object_colors['enemy_missile'], **params)
        
        enemy_missiles = []
        for i, coords in enumerate(coords_list):
            missile = GameObject('enemy_missile', coords, f'enemy_missile_{i}')
            enemy_missiles.append(missile)
        
        # Remove missiles that overlap with divers
        missiles_to_remove = []
        for missile in enemy_missiles:
            for diver in divers:
                if self._objects_overlap(missile, diver):
                    missiles_to_remove.append(missile)
                    break
        
        for missile in missiles_to_remove:
            if missile in enemy_missiles:
                enemy_missiles.remove(missile)
        
        # Add small divers that are actually enemy missiles (Seaquest-specific logic)
        for diver in divers:
            if (6 <= diver.width <= 8) and diver.height == 4:
                missile = GameObject('enemy_missile', diver.bounding_box, 
                                  f'enemy_missile_{len(enemy_missiles)}')
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
            detected_objects['player_missile'] = cleaned_missiles
    
    def _objects_overlap(self, obj1: GameObject, obj2: GameObject) -> bool:
        """Check if two game objects overlap (Seaquest-specific logic)."""
        return (obj1.left >= obj2.left and obj1.left <= obj2.right and 
                obj1.top >= obj2.top and obj1.top <= obj2.bottom)
