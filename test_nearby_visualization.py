#!/usr/bin/env python3
"""
Test script demonstrating nearby relationship visualization with green lines.
"""

import numpy as np
import cv2
from core.game_object import GameObject
from core.relationship_analyzer import BaseRelationshipAnalyzer

def create_test_scene():
    """Create a test scene with objects at different distances."""
    # Create a simple test image
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Create test objects
    detected_objects = {
        'player': [GameObject('player', (50, 100, 20, 20), object_id='player_1')],
        'enemy': [
            GameObject('enemy', (55, 105, 15, 15), object_id='enemy_1'),  # Very close (nearby)
            GameObject('enemy', (80, 110, 15, 15), object_id='enemy_2'),  # Close (nearby)
            GameObject('enemy', (150, 100, 15, 15), object_id='enemy_3'), # Far (not nearby)
        ],
        'collectible': [
            GameObject('collectible', (45, 95, 10, 10), object_id='item_1'),  # Very close (nearby)
        ]
    }
    
    return image, detected_objects

def test_nearby_relationships():
    """Test the nearby relationship functionality."""
    print("Testing nearby relationship visualization...")
    
    # Create test scene
    image, detected_objects = create_test_scene()
    
    # Initialize components
    analyzer = BaseRelationshipAnalyzer()
    visualizer = VisualizationManager()
    
    # Analyze relationships
    relationships = analyzer.analyze_all_relationships(detected_objects)
    connection_list = analyzer.create_connection_list(relationships)
    
    # Print detected relationships
    print("\nDetected relationships:")
    for relationship in relationships:
        print(f"  {relationship}")
    
    # Print connection list
    print("\nConnection list:")
    for connection in connection_list:
        obj1_id = connection['obj1'].object_id
        obj2_id = connection['obj2'].object_id
        rels = connection['relationships']
        distance = connection['distance']
        print(f"  {obj1_id} -> {obj2_id}: {rels} (distance: {distance:.2f})")
    
    # Create visualization
    annotated_image = visualizer.draw_all_objects(image, detected_objects)
    annotated_image = visualizer.draw_relationships(annotated_image, connection_list)
    annotated_image = visualizer.add_object_labels(annotated_image, detected_objects)
    
    # Scale up for better visibility
    scale_factor = 4
    height, width = annotated_image.shape[:2]
    scaled_image = cv2.resize(annotated_image, (width * scale_factor, height * scale_factor), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Display the result
    print(f"\nVisualization created! Green lines indicate nearby relationships.")
    print("The test shows:")
    print("- Player (green box) at center")
    print("- Enemy objects at various distances (red boxes)")
    print("- Collectible item (magenta box)")
    print("- Green lines connect nearby objects (within threshold distance)")
    print("- Black lines connect objects with other relationships")
    
    # Save the visualization
    cv2.imwrite('nearby_relationship_test.png', scaled_image)
    print("Visualization saved as 'nearby_relationship_test.png'")
    
    return scaled_image, relationships, connection_list

if __name__ == "__main__":
    # Run the test
    test_image, relationships, connections = test_nearby_relationships()
    
    print(f"\nTest completed! Found {len(relationships)} total relationships.")
    nearby_count = sum(1 for rel in relationships if rel.relationship_type == 'nearby')
    print(f"Found {nearby_count} nearby relationships that will be drawn with green lines.")
