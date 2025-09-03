#!/usr/bin/env python3
"""
Standalone visualization manager for testing nearby relationships without external dependencies.
"""

import numpy as np
import cv2
from core.game_object import GameObject
from core.relationship_analyzer import BaseRelationshipAnalyzer
from core.config import BASE_VISUALIZATION_COLORS, DEFAULT_OBJECT_COLORS

def draw_bounding_box(image, bbox, color=(255, 255, 255)):
    """Draw a bounding box on the image."""
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

class StandaloneVisualizationManager:
    """Simple visualization manager without external dependencies."""
    
    def __init__(self):
        self.base_colors = BASE_VISUALIZATION_COLORS.copy()
        self.object_colors = DEFAULT_OBJECT_COLORS.copy()
    
    def draw_all_objects(self, image, detected_objects):
        """Draw bounding boxes around all detected objects."""
        annotated_image = image.copy()
        
        for object_type, objects in detected_objects.items():
            color = self.object_colors.get(object_type, (255, 255, 255))
            
            for game_object in objects:
                draw_bounding_box(annotated_image, game_object.bounding_box, color)
                
                # Add object label
                label_pos = (game_object.x, max(game_object.y - 5, 10))
                cv2.putText(annotated_image, game_object.object_id, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_image
    
    def draw_relationships(self, image, connection_list):
        """Draw lines representing relationships between objects."""
        annotated_image = image.copy()
        
        for connection in connection_list:
            obj1 = connection['obj1']
            obj2 = connection['obj2']
            relationships = connection['relationships']
            
            # Calculate centers of both objects
            center1 = obj1.center
            center2 = obj2.center
            
            # Determine line color based on relationship type
            line_color = self.base_colors.get('relationship_line', (0, 0, 0))  # Default black
            
            # Use green line for nearby relationships
            if 'nearby' in relationships:
                line_color = self.base_colors.get('nearby_line', (0, 255, 0))  # Green
                print(f"Drawing GREEN line between {obj1.object_id} and {obj2.object_id} (nearby relationship)")
            else:
                print(f"Drawing BLACK line between {obj1.object_id} and {obj2.object_id} (relationships: {relationships})")
            
            # Draw line between centers
            cv2.line(annotated_image, center1, center2, line_color, 2)
            
            # Create relationship text
            relationships_text = '-'.join([rel[:3].upper() for rel in relationships])
            
            # Calculate midpoint for text placement
            mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
            
            # Draw relationship text
            text_color = (255, 255, 255)  # White text for visibility
            cv2.putText(annotated_image, relationships_text, mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        
        return annotated_image

def create_test_scene():
    """Create a test scene with objects at different distances."""
    # Create a test image (black background)
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Create test objects with specific positions to test nearby relationships
    detected_objects = {
        'player': [GameObject('player', (200, 200, 40, 40), object_id='player_1')],
        'enemy': [
            # Very close to player (should be nearby - distance ~7)
            GameObject('enemy', (205, 205, 30, 30), object_id='enemy_close'),  
            # Medium distance (should be nearby - distance ~20, but within threshold)
            GameObject('enemy', (170, 180, 30, 30), object_id='enemy_medium'),  
            # Far from player (should NOT be nearby - distance ~100)
            GameObject('enemy', (300, 200, 30, 30), object_id='enemy_far'),     
        ],
        'collectible': [
            # Very close to player (should be nearby)
            GameObject('collectible', (195, 195, 20, 20), object_id='item_close'),  
        ]
    }
    
    return image, detected_objects

def test_nearby_relationships():
    """Test the nearby relationship functionality with detailed debugging."""
    print("=== TESTING NEARBY RELATIONSHIP VISUALIZATION ===")
    
    # Create test scene
    image, detected_objects = create_test_scene()
    
    # Print object positions for debugging
    print("\n--- OBJECT POSITIONS ---")
    for obj_type, objects in detected_objects.items():
        for obj in objects:
            print(f"{obj.object_id}: center={obj.center}, bbox={obj.bounding_box}")
    
    # Initialize components
    analyzer = BaseRelationshipAnalyzer()
    visualizer = StandaloneVisualizationManager()
    
    # Analyze relationships
    relationships = analyzer.analyze_all_relationships(detected_objects)
    connection_list = analyzer.create_connection_list(relationships)
    
    # Print all detected relationships
    print("\n--- ALL DETECTED RELATIONSHIPS ---")
    for i, relationship in enumerate(relationships):
        print(f"{i+1}. {relationship}")
    
    # Print connection list with distances
    print("\n--- CONNECTION LIST ---")
    for i, connection in enumerate(connection_list):
        obj1_id = connection['obj1'].object_id
        obj2_id = connection['obj2'].object_id
        rels = connection['relationships']
        distance = connection.get('distance', 'N/A')
        print(f"{i+1}. {obj1_id} -> {obj2_id}: {rels} (distance: {distance:.2f})")
    
    # Check specifically for nearby relationships
    nearby_relationships = [rel for rel in relationships if rel.relationship_type == 'nearby']
    print(f"\n--- NEARBY RELATIONSHIPS FOUND: {len(nearby_relationships)} ---")
    for rel in nearby_relationships:
        print(f"  {rel.obj1.object_id} <-> {rel.obj2.object_id} (distance: {rel.distance:.2f})")
    
    # Create visualization
    print("\n--- CREATING VISUALIZATION ---")
    annotated_image = visualizer.draw_all_objects(image, detected_objects)
    annotated_image = visualizer.draw_relationships(annotated_image, connection_list)
    
    # Scale up for better visibility
    scale_factor = 2
    height, width = annotated_image.shape[:2]
    scaled_image = cv2.resize(annotated_image, (width * scale_factor, height * scale_factor), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Save the visualization
    cv2.imwrite('nearby_relationship_debug.png', scaled_image)
    print(f"\nVisualization saved as 'nearby_relationship_debug.png'")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total relationships found: {len(relationships)}")
    print(f"Nearby relationships found: {len(nearby_relationships)}")
    print(f"Connections to visualize: {len(connection_list)}")
    
    if len(nearby_relationships) == 0:
        print("⚠️  WARNING: No nearby relationships detected!")
        print("   This could be due to:")
        print("   1. Distance threshold is too small")
        print("   2. Objects are too far apart")
        print("   3. Issue with the nearby() function")
    else:
        print("✓ Nearby relationships detected successfully!")
        print("  Green lines should appear in the visualization.")
    
    return scaled_image, relationships, connection_list

if __name__ == "__main__":
    test_nearby_relationships()
