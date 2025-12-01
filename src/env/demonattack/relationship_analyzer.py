"""
DemonAttack-specific relationship analyzer implementation.
"""
from typing import Dict, List, Callable
import sys
import os

sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/')
sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/src')

from core.relationship_analyzer import BaseRelationshipAnalyzer
from core.game_object import SpatialRelationship, GameObject


class DemonAttackRelationshipConfig:
    """Relationship configuration for DemonAttack."""
    
    def __init__(self):
        self.nearby_threshold = 2000  # Square units for nearbyMissile
        self.directly_above_threshold = 20  # Horizontal tolerance for "directly above"
    
    def get_reference_levels(self) -> Dict[str, int]:
        return {}
    
    def get_relationship_rules(self) -> List[Callable]:
        return []
    
    def format_relationship_description(self, relationship: SpatialRelationship) -> str:
        """Format a relationship into a human-readable description."""
        obj1_id = relationship.obj1.object_id
        rel_type = relationship.relationship_type
        
        # For visible relationships (obj2 is a virtual object)
        if hasattr(relationship.obj2, 'object_id') and relationship.obj2.object_id == 'visible_occurrence':
            return f"{rel_type}({obj1_id})"
        
        # For normal two-object relationships
        if relationship.obj2:
            obj2_id = relationship.obj2.object_id
            return f"{rel_type}({obj2_id})"
        
        return f"{rel_type}({obj1_id})"


class DemonAttackRelationshipAnalyzer(BaseRelationshipAnalyzer):
    """DemonAttack-specific relationship analyzer."""
    
    def __init__(self):
        super().__init__(DemonAttackRelationshipConfig())
        self.nearby_threshold = 2000  # Square units
    
    def analyze_all_relationships(self, detected_objects):
        """
        Analyze all relationships in the current frame.
        
        Args:
            detected_objects: Dictionary of detected objects by type
            
        Returns:
            List of SpatialRelationship objects
        """
        relationships = []
        
        # Get player (reference point for most relationships)
        players = detected_objects.get('player', [])
        if not players:
            return relationships
        player = players[0]
        
        # Get all object types
        enemies = detected_objects.get('enemy', [])
        hostile_missiles = detected_objects.get('projectile_hostile', [])
        friendly_missiles = detected_objects.get('projectile_friendly', [])
        
        # Analyze visible relationships (for all occurrences)
        relationships.extend(self._analyze_visible_objects(enemies, hostile_missiles, friendly_missiles))
        
        # Analyze fireReady relationships (friendly missiles inside player bbox)
        relationships.extend(self._analyze_fire_ready(player, friendly_missiles))
        
        # Analyze enemy relationships with player
        for enemy in enemies:
            enemy_relationships = self._analyze_player_enemy_relationships(player, enemy)
            relationships.extend(enemy_relationships)
        
        # Analyze hostile missile relationships with player
        for missile in hostile_missiles:
            missile_relationships = self._analyze_player_missile_relationships(player, missile)
            relationships.extend(missile_relationships)
        
        return relationships
    
    def _analyze_visible_objects(self, enemies, hostile_missiles, friendly_missiles):
        """
        Create visible relationships for all detected objects.
        
        Returns:
            List of SpatialRelationship objects for visible objects
        """
        relationships = []
        
        # Create a virtual "visible occurrence" object to satisfy SpatialRelationship requirements
        virtual_obj = GameObject('visible_occurrence', (0, 0, 1, 1), object_id='visible_occurrence')
        
        # visibleEnemy(enemy) - for all enemies
        for enemy in enemies:
            relationships.append(SpatialRelationship(enemy, virtual_obj, 'visibleEnemy'))
        
        # visibleEnemyMissile(emissile) - for all hostile missiles
        for missile in hostile_missiles:
            relationships.append(SpatialRelationship(missile, virtual_obj, 'visibleEnemyMissile'))
        
        # visiblePlayerMissile(pmissile) - for all friendly missiles
        for missile in friendly_missiles:
            relationships.append(SpatialRelationship(missile, virtual_obj, 'visiblePlayerMissile'))
        
        return relationships
    
    def _analyze_fire_ready(self, player, friendly_missiles):
        """
        Analyze fireReady relationship - when friendly missile is inside player's bounding box.
        This indicates the player is ready to fire or has just fired.
        
        Args:
            player: Player GameObject
            friendly_missiles: List of friendly missile GameObjects
            
        Returns:
            List of SpatialRelationship objects
        """
        relationships = []
        
        for missile in friendly_missiles:
            # Check if missile is inside player's bounding box
            if self._is_inside_bbox(missile, player):
                relationships.append(SpatialRelationship(player, missile, 'fireReady'))
        
        return relationships
    
    def _is_inside_bbox(self, obj1, obj2):
        """
        Check if obj1 is completely inside obj2's bounding box.
        
        Args:
            obj1: GameObject to check if inside
            obj2: GameObject bounding box to check against
            
        Returns:
            bool: True if obj1 is inside obj2's bbox
        """
        # Get bounding box coordinates
        obj1_left = obj1.left
        obj1_right = obj1.right
        obj1_top = obj1.top
        obj1_bottom = obj1.bottom
        
        obj2_left = obj2.left
        obj2_right = obj2.right
        obj2_top = obj2.top
        obj2_bottom = obj2.bottom
        
        # Check if obj1 is completely inside obj2
        is_inside = (obj1_left >= obj2_left and 
                    obj1_right <= obj2_right and 
                    obj1_top >= obj2_top and 
                    obj1_bottom <= obj2_bottom)
        
        return is_inside
    
    def _analyze_player_enemy_relationships(self, player, enemy):
        """
        Analyze all relationships between player and an enemy.
        
        Returns:
            List of SpatialRelationship objects
        """
        relationships = []
        
        player_x, player_y = player.center
        enemy_x, enemy_y = enemy.center
        
        # 1. enemyDirectlyAbove(enemy) - enemy is directly above player (X-axis alignment)
        # Check if enemy's X overlaps with player's bounding box
        if self._is_directly_above(player, enemy):
            relationships.append(SpatialRelationship(player, enemy, 'enemyDirectlyAbove'))
        
        # 2. rightOfEnemy(enemy) - player is to the right of enemy
        # This means enemy is to the left of player
        elif enemy_x < player_x:
            relationships.append(SpatialRelationship(player, enemy, 'rightOfEnemy'))
        
        # 3. leftOfEnemy(enemy) - player is to the left of enemy
        # This means enemy is to the right of player
        elif enemy_x > player_x:
            relationships.append(SpatialRelationship(player, enemy, 'leftOfEnemy'))
        
        return relationships
    
    def _analyze_player_missile_relationships(self, player, missile):
        """
        Analyze all relationships between player and a hostile missile.
        
        Returns:
            List of SpatialRelationship objects
        """
        relationships = []
        
        player_x, player_y = player.center
        missile_x, missile_y = missile.center
        
        # 1. enemyMissileDirectlyAbove(emissile) - missile is directly above player (X-axis alignment)
        # Check if missile's X overlaps with player's bounding box
        if self._is_directly_above(player, missile):
            relationships.append(SpatialRelationship(player, missile, 'enemyMissileDirectlyAbove'))
        
        # 2. rightOfMissile(emissile) - player is to the right of missile
        # This means missile is to the left of player
        elif missile_x < player_x:
            relationships.append(SpatialRelationship(player, missile, 'rightOfMissile'))
        
        # 3. leftOfMissile(emissile) - player is to the left of missile
        # This means missile is to the right of player
        elif missile_x > player_x:
            relationships.append(SpatialRelationship(player, missile, 'leftOfMissile'))
        
        # 4. nearbyMissile(missile) - missile is within 2000 square units of player
        distance_sq = self._calculate_distance_squared(player, missile)
        if distance_sq <= self.nearby_threshold:
            relationships.append(SpatialRelationship(player, missile, 'nearbyMissile'))
        
        return relationships
    
    def _is_directly_above(self, player, obj):
        """
        Check if object is directly above player.
        Object is directly above if:
        1. Object's Y position is above player (obj.y < player.y in screen coordinates)
        2. Object's X range overlaps with player's X range
        """
        obj_x, obj_y = obj.center
        player_x, player_y = player.center
        
        # Object must be above player (lower Y value in screen coordinates)
        if obj_y >= player_y:
            return False
        
        # Check X overlap: object's bounding box overlaps with player's bounding box in X direction
        # Overlap exists if: NOT(obj.right < player.left OR obj.left > player.right)
        x_overlap = not (obj.right < player.left or obj.left > player.right)
        
        return x_overlap
    
    def _calculate_distance_squared(self, obj1, obj2):
        """Calculate squared distance between two objects' centers."""
        x1, y1 = obj1.center
        x2, y2 = obj2.center
        return (x1 - x2)**2 + (y1 - y2)**2
    
    def format_relationships_for_dataframe(self, relationships):
        """
        Format relationships for storage in a pandas DataFrame.
        
        Returns:
            Formatted string with relationships separated by " , "
        """
        formatted_relationships = []
        
        for relationship in relationships:
            rel_type = relationship.relationship_type
            
            # For visible relationships (obj2 is virtual)
            if hasattr(relationship.obj2, 'object_id') and relationship.obj2.object_id == 'visible_occurrence':
                obj1_id = relationship.obj1.object_id
                formatted_relationships.append(f"{rel_type}({obj1_id})")
            # For normal two-object relationships
            elif relationship.obj2:
                obj2_id = relationship.obj2.object_id
                formatted_relationships.append(f"{rel_type}({obj2_id})")
            else:
                # Fallback
                obj1_id = relationship.obj1.object_id
                formatted_relationships.append(f"{rel_type}({obj1_id})")
        
        return " , ".join(formatted_relationships) + " , " if formatted_relationships else ""
