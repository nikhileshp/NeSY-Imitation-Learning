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

        if obj2_type == "facing_side":
            return f"{rel_type}({obj1_type})."
        
        # Special formatting for diver count relationships
        if obj2_type == 'diver_state':
            if rel_type == 'diversfull':
                return "diversfull(player)."
            elif rel_type == 'diversNotfull':
                return "diversNotfull(player)."
     
        # Special formatting for specific Seaquest object types
        if obj2_type == 'enemy':
            return f"{rel_type}Enemy({obj1_type}, {obj2_id})."
        elif obj2_type == 'enemy_submarine':
            return f"{rel_type}Enemy({obj1_type}, {obj2_id})."
        elif obj2_type == 'player_missile':
            return f"{rel_type}Missile({obj1_type}, {obj2_id})."
        elif obj2_type == 'enemy_missile':
            return f"{rel_type}Missile({obj1_type}, {obj2_id})."
        elif obj2_type == 'diver':
            return f"{rel_type}Diver({obj1_type}, {obj2_id})."
        
        else:
            return f"{rel_type}({obj1_type}, {obj2_id})."

        
class SeaquestRelationshipAnalyzer(BaseRelationshipAnalyzer):
    """Seaquest-specific relationship analyzer."""
    
    def __init__(self):
        """Initialize with Seaquest relationship configuration."""
        super().__init__(SeaquestRelationshipConfig())
        # State tracking for diver count hysteresis to handle blinking divers
        self._previous_diver_state = 'diversNotfull'  # Track previous state
        self._diver_full_threshold = 6  # Threshold to become diversfull
        self._diver_not_full_threshold = 5  # Threshold to become diversNotfull
    
    def analyze_all_relationships(self, detected_objects):
        """
        Analyze only specific relationships for Seaquest:
        - Facing side of certain objects
        - Player vs water surface
        - Player vs enemies 
        - Player vs enemy submarines
        - Player vs enemy missiles (for cleanup/filtering)
        - Player vs divers

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
            # print(reference_levels)
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


        # Return facing side of objects that contain that characteristic
        for obj in detected_objects.values():
            for o in obj:
                
                if 'facing_side' in o.characteristics:
                    print("Found facing side characteristic", o.characteristics)
                    char_relationship = self._analyze_object_characteristics_relationship(o)
                    if char_relationship:
                        relationships.append(char_relationship)

        # Analyze diver count relationships (diversfull/diversNotfull)
        diver_count_relationship = self._analyze_diver_count_relationship(detected_objects, player)
        if diver_count_relationship:
            relationships.append(diver_count_relationship)

        return relationships
    
    def _analyze_diver_count_relationship(self, detected_objects, player):
        """
        Analyze diver count to determine diversfull or diversNotfull relationship.
        Uses hysteresis to handle blinking divers - once diversfull is reached (6+ divers),
        it remains diversfull until the count drops to 5 or below.
        
        Args:
            detected_objects: Dictionary mapping object types to lists of GameObjects
            player: Player GameObject
            
        Returns:
            SpatialRelationship object or None
        """
        from core.game_object import GameObject
        
        # Count collected divers
        collected_divers = detected_objects.get('collected_diver', [])
        collected_diver_count = len(collected_divers)
        
        # Create a virtual object to represent the diver count state
        virtual_diver_state = GameObject('diver_state', (0, 0, 0, 0), object_id='diver_count_state')
        
        # Hysteresis logic to handle blinking divers:
        # - Transition to diversfull when count >= 6
        # - Transition to diversNotfull only when count is consistently <= 5 (not during blinking)
        # - During blinking (count oscillates 6↔0), maintain diversfull state
        
        current_state = self._previous_diver_state
        
        if self._previous_diver_state == 'diversNotfull':
            # Currently not full - check if we should transition to full
            if collected_diver_count >= self._diver_full_threshold:
                current_state = 'diversfull'
                self._previous_diver_state = 'diversfull'
        else:  # self._previous_diver_state == 'diversfull'
            # Currently full - only transition to not full if count is stable at <= 5
            # During blinking, count oscillates between 6 and 0, so we ignore count=0 drops
            # Only transition when count is consistently between 1-5 (actual diver loss)
            if 1 <= collected_diver_count <= self._diver_not_full_threshold:
                current_state = 'diversNotfull'
                self._previous_diver_state = 'diversNotfull'
            # If count is 0 (blinking) or >= 6 (still full), maintain diversfull state
        
        # Return the appropriate relationship
        return SpatialRelationship(player, virtual_diver_state, current_state)
    
    def format_relationships_for_dataframe(self, relationships):
        """
        Format relationships for storage in a pandas DataFrame with Seaquest-specific formatting.
        
        Args:
            relationships: List of SpatialRelationship objects
            
        Returns:
            Formatted string of relationships
        """
        formatted_relationships = []
        
        for relationship in relationships:
            obj1_type = relationship.obj1.object_type
            obj2_id = relationship.obj2.object_id
            obj2_type = relationship.obj2.object_type
            rel_type = relationship.relationship_type
            
            # Special handling for water surface relationships
            if any(level in relationship.obj2.object_type.lower() 
                   for level in ['water', 'surface', 'ground', 'ceiling']):
                formatted_relationships.append(f"{rel_type}({obj1_type})")
            # Special handling for diver count relationships
            elif obj2_type == 'diver_state':
                formatted_relationships.append(f"{rel_type}({obj1_type})")
            # Special handling for facing side relationships
            elif obj2_type == 'facing_side':
                formatted_relationships.append(f"{rel_type}({obj1_type})")
            else:
                formatted_relationships.append(f"{rel_type}({obj1_type},{obj2_id})")
        
        return " , ".join(formatted_relationships) + " , " if formatted_relationships else ""
