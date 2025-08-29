"""
Core relationship analyzer module for analyzing spatial relationships between game objects.
"""
from typing import List, Dict, Tuple, Optional, Protocol, Callable
from .game_object import GameObject, SpatialRelationship, above_reference_level, right_of, left_of, above_of, below_of, same_level_of


class GameRelationshipConfig(Protocol):
    """Protocol for game-specific relationship configuration."""
    
    def get_reference_levels(self) -> Dict[str, int]:
        """Return dictionary of reference levels (e.g., water_surface: 47)."""
        ...
    
    def get_relationship_rules(self) -> List[Callable]:
        """Return list of custom relationship analysis functions."""
        ...
    
    def format_relationship_description(self, relationship: SpatialRelationship) -> str:
        """Format relationship for human-readable output."""
        ...


class BaseRelationshipAnalyzer:
    """Analyzes spatial relationships between game objects."""
    
    def __init__(self, game_config: Optional[GameRelationshipConfig] = None):
        """
        Initialize the relationship analyzer.
        
        Args:
            game_config: Optional game-specific configuration for relationships
        """
        self.game_config = game_config
    
    def analyze_all_relationships(self, detected_objects: Dict[str, List[GameObject]]) -> List[SpatialRelationship]:
        """
        Analyze all spatial relationships between detected objects.
        
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
        
        # Analyze reference level relationships (e.g., water surface)
        if self.game_config:
            reference_levels = self.game_config.get_reference_levels()
            for level_name, level_y in reference_levels.items():
                ref_relationship = self._analyze_reference_level_relationship(player, level_name, level_y)
                if ref_relationship:
                    relationships.append(ref_relationship)
        
        # Analyze relationships with other objects
        for object_type, objects in detected_objects.items():
            if object_type == 'player':
                continue
                
            for obj in objects:
                obj_relationships = self._analyze_object_relationships(player, obj)
                relationships.extend(obj_relationships)
        
        # Apply custom relationship rules if available
        if self.game_config:
            custom_rules = self.game_config.get_relationship_rules()
            for rule in custom_rules:
                custom_relationships = rule(detected_objects)
                relationships.extend(custom_relationships)
        
        return relationships
    
    def _analyze_reference_level_relationship(self, player: GameObject, level_name: str, level_y: int) -> Optional[SpatialRelationship]:
        """
        Analyze the relationship between player and a reference level.
        
        Args:
            player: Player GameObject
            level_name: Name of the reference level
            level_y: Y-coordinate of the reference level
            
        Returns:
            SpatialRelationship object or None
        """
        if above_reference_level(player, level_y):
            # Create a virtual reference level object
            ref_obj = GameObject(level_name, (0, level_y, 160, 1), level_name)
            return SpatialRelationship(player, ref_obj, f'above{level_name.capitalize()}')
        else:
            ref_obj = GameObject(level_name, (0, level_y, 160, 1), level_name)
            return SpatialRelationship(player, ref_obj, f'below{level_name.capitalize()}')
    
    def _analyze_object_relationships(self, obj1: GameObject, obj2: GameObject) -> List[SpatialRelationship]:
        """
        Analyze all possible relationships between two objects.
        
        Args:
            obj1: First GameObject
            obj2: Second GameObject
            
        Returns:
            List of SpatialRelationship objects
        """
        relationships = []
        
        # Horizontal relationships
        if left_of(obj1, obj2):
            relationships.append(SpatialRelationship(obj1, obj2, 'leftOf'))
        elif right_of(obj1, obj2):
            relationships.append(SpatialRelationship(obj1, obj2, 'rightOf'))
        
        # Vertical relationships
        if above_of(obj1, obj2):
            relationships.append(SpatialRelationship(obj1, obj2, 'aboveOf'))
        elif below_of(obj1, obj2):
            relationships.append(SpatialRelationship(obj1, obj2, 'belowOf'))
        else:
            # Check if they are at the same level
            if same_level_of(obj1, obj2):
                relationships.append(SpatialRelationship(obj1, obj2, 'sameLevelAs'))
        
        return relationships
    
    def create_connection_list(self, relationships: List[SpatialRelationship]) -> List[Dict]:
        """
        Create a connection list that groups relationships between the same pair of objects.
        
        Args:
            relationships: List of SpatialRelationship objects
            
        Returns:
            List of connection dictionaries
        """
        connection_list = []
        
        for relationship in relationships:
            # Skip reference level relationships for connection list
            if any(level in relationship.obj2.object_id.lower() 
                   for level in ['water', 'surface', 'ground', 'ceiling']):
                continue
            
            # Check if we already have a connection between these objects
            found = False
            for connection in connection_list:
                if (connection['obj1'] == relationship.obj1 and 
                    connection['obj2'] == relationship.obj2):
                    connection['relationships'].append(relationship.relationship_type)
                    found = True
                    break
            
            if not found:
                connection = {
                    'obj1': relationship.obj1,
                    'obj2': relationship.obj2,
                    'relationships': [relationship.relationship_type],
                    'distance': relationship.distance
                }
                connection_list.append(connection)
        
        return connection_list
    
    def get_relationship_descriptions(self, relationships: List[SpatialRelationship]) -> List[str]:
        """
        Get human-readable descriptions of relationships.
        
        Args:
            relationships: List of SpatialRelationship objects
            
        Returns:
            List of relationship description strings
        """
        descriptions = []
        
        for relationship in relationships:
            if self.game_config:
                # Use game-specific formatting if available
                description = self.game_config.format_relationship_description(relationship)
            else:
                # Use default formatting
                description = self._default_format_relationship(relationship)
            
            descriptions.append(description)
        
        return descriptions
    
    def _default_format_relationship(self, relationship: SpatialRelationship) -> str:
        """Default relationship formatting."""
        obj1_type = relationship.obj1.object_type
        obj2_type = relationship.obj2.object_type
        rel_type = relationship.relationship_type
        
        if any(level in obj2_type.lower() for level in ['water', 'surface', 'ground', 'ceiling']):
            return f"{rel_type}({obj1_type})."
        else:
            obj1_id = relationship.obj1.object_id
            obj2_id = relationship.obj2.object_id
            return f"{rel_type}({obj1_type}, {obj2_id})."
    
    def format_relationships_for_dataframe(self, relationships: List[SpatialRelationship]) -> str:
        """
        Format relationships for storage in a pandas DataFrame.
        
        Args:
            relationships: List of SpatialRelationship objects
            
        Returns:
            Formatted string of relationships
        """
        formatted_relationships = []
        
        for relationship in relationships:
            obj1_type = relationship.obj1.object_type
            obj2_id = relationship.obj2.object_id
            rel_type = relationship.relationship_type
            
            if any(level in relationship.obj2.object_type.lower() 
                   for level in ['water', 'surface', 'ground', 'ceiling']):
                formatted_relationships.append(f"{rel_type}({obj1_type})")
            else:
                formatted_relationships.append(f"{rel_type}({obj1_type},{obj2_id})")
        
        return " , ".join(formatted_relationships) + " , " if formatted_relationships else ""
