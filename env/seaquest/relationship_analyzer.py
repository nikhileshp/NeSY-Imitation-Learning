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
        obj1_id = relationship.obj1.object_id
        obj2_type = relationship.obj2.object_type
        obj2_id = relationship.obj2.object_id
        rel_type = relationship.relationship_type
        
        # Special formatting for water surface relationships - add empty parentheses
        if obj2_type == 'water_surface':
            if rel_type == 'aboveWater_surface':
                return "aboveWater()."
            else:
                return "belowWater()."

        # Special formatting for facing side relationships - include enemy argument for enemy submarines
        if obj2_type == "facing_side":
            if obj1_type == 'enemy_submarine':
                return f"{rel_type}({obj1_id})."
            else:
                return f"{rel_type}()."
        
        # Special formatting for diver count relationships - add empty parentheses
        if obj2_type == 'diver_state':
            if rel_type == 'diversfull':
                return "diversfull()."
            elif rel_type == 'diversNotfull':
                return "diversNotfull()."
            elif rel_type == 'diversEmpty':
                return "diversEmpty()."
        
        # Special formatting for oxygen relationships - add empty parentheses
        if obj2_type == 'oxygen_state':
            if rel_type == 'oxygenLow':
                return "oxygenLow()."
            elif rel_type == 'oxygenOk':
                return "oxygenOk()."
     
        # Special formatting for specific Seaquest object types - keep second argument
        if obj2_type == 'enemy':
            return f"{rel_type}Enemy({obj2_id})."
        elif obj2_type == 'enemy_submarine':
            return f"{rel_type}Enemy({obj2_id})."
        elif obj2_type == 'player_missile':
            return f"{rel_type}Missile({obj2_id})."
        elif obj2_type == 'enemy_missile':
            return f"{rel_type}Missile({obj2_id})."
        elif obj2_type == 'diver':
            return f"{rel_type}Diver({obj2_id})."
        
        else:
            return f"{rel_type}({obj2_id})."

        
class SeaquestRelationshipAnalyzer(BaseRelationshipAnalyzer):
    """Seaquest-specific relationship analyzer."""
    
    def __init__(self):
        """Initialize with Seaquest relationship configuration."""
        super().__init__(SeaquestRelationshipConfig())
        # State tracking for diver count hysteresis to handle blinking divers
        self._previous_diver_state = 'diversEmpty'  # Start with empty state
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
        
        # First, analyze facing side characteristics for all objects (regardless of whether players exist)
        for obj in detected_objects.values():
            for o in obj:
                if 'facing_side' in o.characteristics:
                    print("Found facing side characteristic", o.characteristics)
                    char_relationship = self._analyze_object_characteristics_relationship(o)
                    if char_relationship:
                        relationships.append(char_relationship)
        
        # Get player objects for other relationship analysis
        players = detected_objects.get('player', [])
        
        # If no players, return just the facing relationships
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


      

        # Analyze diver count relationships (diversfull/diversNotfull)
        diver_count_relationship = self._analyze_diver_count_relationship(detected_objects, player)
        if diver_count_relationship:
            relationships.append(diver_count_relationship)
        
        # Analyze oxygen level relationships (oxygenLow/oxygenOk)
        oxygen_relationship = self._analyze_oxygen_relationship(detected_objects, player)
        if oxygen_relationship:
            relationships.append(oxygen_relationship)

        return relationships
    
    def _analyze_diver_count_relationship(self, detected_objects, player):
        """
        Analyze diver count to determine diversfull, diversNotfull, or diversEmpty relationship.
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
        virtual_diver_state = GameObject('diver_state', (0, 0, 0, 0), {}, object_id='diver_count_state')
        
        # Check for empty divers case first
        if collected_diver_count == 0:
            current_state = 'diversEmpty'
            self._previous_diver_state = 'diversEmpty'  # Reset state when empty
        else:
            # Hysteresis logic to handle blinking divers:
            # - Transition to diversfull when count >= 6
            # - Transition to diversNotfull only when count is consistently <= 5 (not during blinking)
            # - During blinking (count oscillates 6↔0), maintain diversfull state
            
            current_state = self._previous_diver_state
            
            # If previous state was diversEmpty, transition to diversNotfull when divers are collected
            if self._previous_diver_state == 'diversEmpty':
                if collected_diver_count >= self._diver_full_threshold:
                    current_state = 'diversfull'
                    self._previous_diver_state = 'diversfull'
                else:
                    current_state = 'diversNotfull'
                    self._previous_diver_state = 'diversNotfull'
            elif self._previous_diver_state == 'diversNotfull':
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
    
    def _analyze_oxygen_relationship(self, detected_objects, player):
        """
        Analyze oxygen level to determine oxygenLow or oxygenOk relationship.
        Uses the presence of oxygen_bar and oxygen_bar_depleted objects to determine oxygen state.
        
        Args:
            detected_objects: Dictionary mapping object types to lists of GameObjects
            player: Player GameObject
            
        Returns:
            SpatialRelationship object or None
        """
        from core.game_object import GameObject
        
        # Get oxygen-related objects
        oxygen_bars = detected_objects.get('oxygen_bar', [])
        oxygen_depleted = detected_objects.get('oxygen_depleted', [])
        
        # Create a virtual object to represent the oxygen state
        virtual_oxygen_state = GameObject('oxygen_state', (0, 0, 0, 0), object_id='oxygen_level_state')
        
        # Determine oxygen state based on what objects are detected
        # If we have more depleted oxygen than full oxygen bars, oxygen is low
        if len(oxygen_depleted) > len(oxygen_bars):
            oxygen_state = 'oxygenLow'
        else:
            oxygen_state = 'oxygenOk'
        
        # Alternative logic: if oxygen_depleted is detected at all, consider it low
        # This might be more sensitive to early low oxygen warnings
        if oxygen_depleted:
            oxygen_state = 'oxygenLow'
        
        # Return the appropriate relationship
        return SpatialRelationship(player, virtual_oxygen_state, oxygen_state)
    
    def _analyze_object_characteristics_relationship(self, game_object):
        """
        Analyze object characteristics to create facing side relationships.
        For enemy submarines, prioritizes movement-based facing detection.
        
        Args:
            game_object: GameObject with characteristics
            
        Returns:
            SpatialRelationship object or None
        """
        from core.game_object import GameObject
        
        if 'facing_side' not in game_object.characteristics:
            return None
            
        facing_direction = game_object.characteristics['facing_side']
        if not facing_direction:
            return None
        
        # For enemy submarines, check if this is movement-based detection
        facing_source = game_object.characteristics.get('facing_source', 'visual')
        
        # Create virtual object for the facing direction
        virtual_facing_object = GameObject('facing_side', (0, 0, 0, 0), 
                                         object_id=facing_direction)
        
        # Create relationship type based on object type and facing direction
        if game_object.object_type == 'enemy_submarine':
            # Special case for enemy submarines - use enemyFacing prefix
            relationship_type = f"enemyFacing{facing_direction.capitalize()}"
            
            # Add debug info to show the source of facing detection
            if facing_source == 'movement_tracking':
                # This is the preferred, stable movement-based detection
                pass
            else:
                # This is visual-based detection, less reliable
                print(f"Warning: Enemy submarine {game_object.object_id} using visual facing detection")
        else:
            # Standard facing relationship
            relationship_type = f"facing{facing_direction.capitalize()}"
        
        return SpatialRelationship(game_object, virtual_facing_object, relationship_type)
    
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
            obj1_id = relationship.obj1.object_id
            obj2_id = relationship.obj2.object_id
            obj2_type = relationship.obj2.object_type
            rel_type = relationship.relationship_type
            
            # Special handling for water surface relationships - no arguments
            if any(level in relationship.obj2.object_type.lower() 
                   for level in ['water', 'surface', 'ground', 'ceiling']):
                formatted_relationships.append(f"{rel_type}")
            # Special handling for diver count relationships - no arguments
            elif obj2_type == 'diver_state':
                formatted_relationships.append(f"{rel_type}")
            # Special handling for facing side relationships - include enemy argument for enemy submarines
            elif obj2_type == 'facing_side':
                if obj1_type == 'enemy_submarine':
                    formatted_relationships.append(f"{rel_type}({obj1_id})")
                else:
                    formatted_relationships.append(f"{rel_type}")
            # Special handling for oxygen relationships - no arguments
            elif obj2_type == 'oxygen_state':
                formatted_relationships.append(f"{rel_type}")
            else:
                formatted_relationships.append(f"{rel_type}({obj2_id})")
        
        return " , ".join(formatted_relationships) + " , " if formatted_relationships else ""
