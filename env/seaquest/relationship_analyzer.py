"""
Seaquest-specific relationship analyzer implementation.
"""
from typing import Dict, List, Callable
from core.relationship_analyzer import BaseRelationshipAnalyzer
from core.game_object import SpatialRelationship
from .config import WATER_SURFACE_Y


class SeaquestRelationshipConfig:
    """Relationship configuration for Seaquest."""
    
    def get_reference_levels(self) -> Dict[str, int]:
        """Return reference levels for Seaquest (water surface)."""
        return {"water_surface": WATER_SURFACE_Y}
    
    def get_relationship_rules(self) -> List[Callable]:
        """Return custom relationship rules for Seaquest."""
        # No custom rules for now, but this could include game-specific logic
        return []
    
    def format_relationship_description(self, relationship: SpatialRelationship) -> str:
        """Format relationship descriptions in Seaquest style."""
        obj1_type = relationship.obj1.object_type
        obj2_type = relationship.obj2.object_type
        obj2_id = relationship.obj2.object_id
        rel_type = relationship.relationship_type
        
        # Special formatting for water surface relationships
        if obj2_type == 'water_surface':
            if rel_type == 'aboveWater_surface':
                return "aboveWater(player)."
            else:
                return "belowWater(player)."
        
        # Special formatting for specific Seaquest object types
        if obj2_type == 'enemy':
            return f"{rel_type}Enemy({obj1_type}, {obj2_id})."
        elif obj2_type == 'enemy_submarine':
            return f"{rel_type}Enemy({obj1_type}, {obj2_id})."
        elif obj2_type == 'player_missile':
            return f"{rel_type}Missile({obj1_type}, {obj2_id})."
        else:
            return f"{rel_type}({obj1_type}, {obj2_id})."


class SeaquestRelationshipAnalyzer(BaseRelationshipAnalyzer):
    """Seaquest-specific relationship analyzer."""
    
    def __init__(self):
        """Initialize with Seaquest relationship configuration."""
        super().__init__(SeaquestRelationshipConfig())
    
    def analyze_all_relationships(self, detected_objects):
        """
        Analyze only specific relationships for Seaquest:
        - Player vs water surface
        - Player vs enemies 
        - Player vs enemy submarines
        - Player vs player missiles (for cleanup/filtering)
        
        Args:
            detected_objects: Dictionary mapping object types to lists of GameObjects
            
        Returns:
            List of SpatialRelationship objects
        """
        relationships = []
        
        # Get player objects for relationship analysis
        players = detected_objects.get('player', [])
        if not players:
            return relationships
        
        # For now, analyze relationships with the first player
        player = players[0]
        
        # Analyze water surface relationship
        if self.game_config:
            reference_levels = self.game_config.get_reference_levels()
            for level_name, level_y in reference_levels.items():
                ref_relationship = self._analyze_reference_level_relationship(player, level_name, level_y)
                if ref_relationship:
                    relationships.append(ref_relationship)
        
        # Only analyze relationships with specific object types
        relevant_object_types = ['enemy', 'enemy_submarine', 'enemy_missile', 'diver']
        
        for object_type in relevant_object_types:
            objects = detected_objects.get(object_type, [])
            for obj in objects:
                obj_relationships = self._analyze_object_relationships(player, obj)
                relationships.extend(obj_relationships)
        
        return relationships
