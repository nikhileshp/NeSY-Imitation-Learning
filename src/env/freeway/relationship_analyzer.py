"""
Freeway-specific relationship analyzer implementation.
"""

from typing import Dict, List, Callable
import sys
import os

sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/')
sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/src')

from core.relationship_analyzer import BaseRelationshipAnalyzer
from core.game_object import SpatialRelationship, GameObject

try:
    from .config import LANE_POSITIONS
except ImportError:
    try:
        from config import LANE_POSITIONS
    except ImportError:
        from env.freeway.config import LANE_POSITIONS

class FreewayRelationshipConfig:
    """Relationship configuration for Freeway."""
    
    def __init__(self):
        self.lane_positions = LANE_POSITIONS
    
    def get_reference_levels(self) -> Dict[str, int]:
        return {}
    
    def get_relationship_rules(self) -> List[Callable]:
        return []
    
    def format_relationship_description(self, relationship: SpatialRelationship) -> str:
        obj1_id = relationship.obj1.object_id
        obj2_id = relationship.obj2.object_id
        rel_type = relationship.relationship_type
        if rel_type == 'carFacingSide':
            return f"{rel_type}({obj1_id},{obj2_id})"
        else:
            return f"{rel_type}({obj2_id})"

class FreewayRelationshipAnalyzer(BaseRelationshipAnalyzer):
    """Freeway-specific relationship analyzer."""
    
    def __init__(self):
        super().__init__(FreewayRelationshipConfig())
        self.nearby_threshold = 1000
    
    def analyze_all_relationships(self, detected_objects):
        relationships = []
        chickens = detected_objects.get('chicken', [])
        chicken = None
        for c in chickens:
            if hasattr(c, 'characteristics') and c.characteristics.get('player') == 1:
                chicken = c
                break
        
        if not chicken:
            return relationships
        
        # Get all cars
        all_cars = []
        for lane_num in range(1, 11):
            car_type = f'car{lane_num}'
            cars = detected_objects.get(car_type, [])
            all_cars.extend(cars)
        
        for car in all_cars:
            car_relationships = self._analyze_chicken_car_relationships(chicken, car)
            relationships.extend(car_relationships)
        
        return relationships
    
    def _analyze_chicken_car_relationships(self, chicken, car):
        """
        Analyze all relationships between chicken and a car.
        
        Returns:
            List of SpatialRelationship objects
        """
        relationships = []
        chicken_x, chicken_y = chicken.center
        car_x, car_y = car.center

        # 1. carBelow(carid) - car's y position is below chicken's y
        if car_y > chicken_y:
            relationships.append(SpatialRelationship(chicken, car, 'carBelow'))

        # 2. carAbove(carid) - car's y position is above chicken's y
        if car_y < chicken_y:
            relationships.append(SpatialRelationship(chicken, car, 'carAbove'))

        # 3. nearbyCar(carid) - within 4000 sq units
        distance_sq = self._calculate_distance_squared(chicken, car)
        if distance_sq <= self.nearby_threshold:
            relationships.append(SpatialRelationship(chicken, car, 'nearbyCar'))

        # 4. carDirectlyAbove(carid) - car is directly above chicken (X overlap with chicken's bounding box)
        # Check if car's X range overlaps with chicken's X range AND car is above
        if self._is_directly_above(chicken, car):
            relationships.append(SpatialRelationship(chicken, car, 'carDirectlyAbove'))

        # 5. carDirectlyBelow(carid) - car is directly below chicken (X overlap with chicken's bounding box)
        # Check if car's X range overlaps with chicken's X range AND car is below
        if self._is_directly_below(chicken, car):
            relationships.append(SpatialRelationship(chicken, car, 'carDirectlyBelow'))

        # 6. carFacingSide(carid, side) - from car characteristics
        if hasattr(car, 'characteristics') and 'direction' in car.characteristics:
            direction = car.characteristics['direction']
            virtual_direction = GameObject('direction', (0, 0, 0, 0), object_id=direction)
            relationships.append(SpatialRelationship(car, virtual_direction, 'carFacingSide'))

        # 7. sameLevelAsCar(carid) - bounding boxes overlap in Y direction
        if self._is_same_level(chicken, car):
            relationships.append(SpatialRelationship(chicken, car, 'sameLevelAsCar'))

        # 8. leftOfCar(carid) - chicken is to the left of car (X only)
        if chicken_x < car_x:
            relationships.append(SpatialRelationship(chicken, car, 'leftOfCar'))

        # 9. rightOfCar(carid) - chicken is to the right of car (X only)
        if chicken_x > car_x:
            relationships.append(SpatialRelationship(chicken, car, 'rightOfCar'))
        
        return relationships

    def _is_directly_above(self, chicken, car):
        """
        Check if car is directly above chicken.
        Car is directly above if:
        1. Car's Y position is above chicken (car.bottom <= chicken.top)
        2. Car's X range overlaps with chicken's X range
        """
        car_x, car_y = car.center
        chicken_x, chicken_y = chicken.center
        
        # Car must be above chicken
        if car_y >= chicken_y:
            return False
        
        # Check X overlap: car's bounding box overlaps with chicken's bounding box in X direction
        # Overlap exists if: NOT(car.right < chicken.left OR car.left > chicken.right)
        x_overlap = not (car.right < chicken.left or car.left > chicken.right)
        
        return x_overlap

    def _is_directly_below(self, chicken, car):
        """
        Check if car is directly below chicken.
        Car is directly below if:
        1. Car's Y position is below chicken (car.top >= chicken.bottom)
        2. Car's X range overlaps with chicken's X range
        """
        car_x, car_y = car.center
        chicken_x, chicken_y = chicken.center
        
        # Car must be below chicken
        if car_y <= chicken_y:
            return False
        
        # Check X overlap: car's bounding box overlaps with chicken's bounding box in X direction
        x_overlap = not (car.right < chicken.left or car.left > chicken.right)
        
        return x_overlap

    def _calculate_distance_squared(self, obj1, obj2):
        """Calculate squared distance between two objects' centers."""
        x1, y1 = obj1.center
        x2, y2 = obj2.center
        return (x1 - x2)**2 + (y1 - y2)**2

    def _get_lane_for_position(self, y_position):
        """Get lane number for a given Y position."""
        for lane_num, (miny, maxy) in self.game_config.lane_positions.items():
            if miny <= y_position <= maxy:
                return lane_num
        return None

    def _get_lane_for_car(self, car):
        """Get lane number from car's characteristics or position."""
        if hasattr(car, 'characteristics') and 'lane' in car.characteristics:
            return car.characteristics['lane']
        _, car_y = car.center
        return self._get_lane_for_position(car_y)

    def _is_same_level(self, chicken, car):
        """Check if chicken and car bounding boxes overlap in Y direction."""
        return not (chicken.bottom < car.top or chicken.top > car.bottom)

    def format_relationships_for_dataframe(self, relationships):
        """Format relationships for storage in a pandas DataFrame."""
        formatted_relationships = []
        for relationship in relationships:
            obj1_id = relationship.obj1.object_id
            obj2_id = relationship.obj2.object_id
            rel_type = relationship.relationship_type
            if rel_type == 'carFacingSide':
                formatted_relationships.append(f"{rel_type}({obj1_id},{obj2_id})")
            else:
                formatted_relationships.append(f"{rel_type}({obj2_id})")
        return " , ".join(formatted_relationships) + " , " if formatted_relationships else ""
