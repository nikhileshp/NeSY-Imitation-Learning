"""
Core game object classes and spatial relationship functions.
"""
from typing import List, Tuple, Optional
from .config import MIN_COVERAGE_RATIO


class GameObject:
    """Represents a game object with position, size, and type information."""
    
    def __init__(self, object_type: str, bounding_box: Tuple[int, int, int, int], facing_side: Optional[str] = None,
                 object_id: Optional[str] = None):
        """
        Initialize a game object.
        
        Args:
            object_type: Type of the object (e.g., 'player', 'enemy', 'diver')
            bounding_box: Tuple of (x, y, width, height)
            object_id: Optional unique identifier for the object
        """
        self.object_type = object_type
        self.x, self.y, self.width, self.height = bounding_box
        self.facing_side = facing_side
        self.object_id = object_id or f"{object_type}_{id(self)}"
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Returns the bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Returns the center point of the object."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def left(self) -> int:
        """Returns the left x-coordinate."""
        return self.x
    
    @property
    def right(self) -> int:
        """Returns the right x-coordinate."""
        return self.x + self.width
    
    @property
    def top(self) -> int:
        """Returns the top y-coordinate."""
        return self.y
    
    @property
    def bottom(self) -> int:
        """Returns the bottom y-coordinate."""
        return self.y + self.height
    
    def __repr__(self) -> str:
        return f"GameObject({self.object_id}, {self.object_type}, {self.bounding_box})"


class SpatialRelationship:
    """Represents a spatial relationship between two game objects."""
    
    def __init__(self, obj1: GameObject, obj2: GameObject, relationship_type: str):
        """
        Initialize a spatial relationship.
        
        Args:
            obj1: First game object
            obj2: Second game object  
            relationship_type: Type of relationship (e.g., 'leftOf', 'aboveOf')
        """
        self.obj1 = obj1
        self.obj2 = obj2
        self.relationship_type = relationship_type
        self.distance = self._calculate_distance()
    
    def _calculate_distance(self) -> float:
        """Calculate Euclidean distance between object centers."""
        center1 = self.obj1.center
        center2 = self.obj2.center
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    def __repr__(self) -> str:
        return (f"SpatialRelationship({self.obj1.object_id}, {self.obj2.object_id}, "
                f"{self.relationship_type}, distance={self.distance:.2f})")


def above_reference_level(obj: GameObject, reference_y: int) -> bool:
    """Check if an object is above a reference level (e.g., water surface)."""
    return obj.y < reference_y


def right_of(obj1: GameObject, obj2: GameObject) -> bool:
    """Check if obj1 is to the right of obj2."""
    return obj2.right <= obj1.left


def left_of(obj1: GameObject, obj2: GameObject) -> bool:
    """Check if obj1 is to the left of obj2."""
    return obj1.right <= obj2.left


def above_of(obj1: GameObject, obj2: GameObject, min_coverage: float = MIN_COVERAGE_RATIO) -> bool:
    """Check if obj1 is above obj2."""
    mid_y1 = obj1.y + obj1.height // 2
    coverage = max(0, min(obj1.bottom, obj2.bottom) - max(obj1.top, obj2.top))
    coverage_ratio = coverage / obj2.height if obj2.height > 0 else 0
    return mid_y1 < obj2.top and (coverage_ratio < min_coverage)


def below_of(obj1: GameObject, obj2: GameObject, min_coverage: float = MIN_COVERAGE_RATIO) -> bool:
    """Check if obj1 is below obj2."""
    mid_y1 = obj1.y + obj1.height // 2
    coverage = max(0, min(obj1.bottom, obj2.bottom) - max(obj1.top, obj2.top))
    coverage_ratio = coverage / obj2.height if obj2.height > 0 else 0
    return mid_y1 > obj2.bottom and (coverage_ratio < min_coverage)


def same_level_of(obj1: GameObject, obj2: GameObject, min_coverage: float = MIN_COVERAGE_RATIO) -> bool:
    """Check if obj1 is at the same level as obj2."""
    mid_y1 = obj1.y + obj1.height // 2
    coverage = max(0, min(obj1.bottom, obj2.bottom) - max(obj1.top, obj2.top))
    coverage_ratio = coverage / obj2.height if obj2.height > 0 else 0
    return (obj2.top <= mid_y1 <= obj2.bottom) and (coverage_ratio >= min_coverage)


def overlaps(obj1: GameObject, obj2: GameObject) -> bool:
    """Check if two objects overlap."""
    return not (obj1.right <= obj2.left or obj2.right <= obj1.left or 
                obj1.bottom <= obj2.top or obj2.bottom <= obj1.top)


def contains(obj1: GameObject, obj2: GameObject) -> bool:
    """Check if obj1 contains obj2."""
    return (obj1.left <= obj2.left and obj1.right >= obj2.right and
            obj1.top <= obj2.top and obj1.bottom >= obj2.bottom)
