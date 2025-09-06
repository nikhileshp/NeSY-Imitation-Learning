#!/usr/bin/env python3
"""
Test script demonstrating enhanced visualization features:
1. Object indexes on top-right corner of bounding boxes
2. Directional arrows for objects with facing side relationships
"""

import numpy as np
import cv2
from core.game_object import GameObject, SpatialRelationship
from core.visualization_manager import VisualizationManager
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer

def create_test_scene_with_facing_objects():
    """Create a test scene with objects that have facing side relationships."""
    # Create a test image (dark blue background)
    image = np.full((400, 600, 3), (40, 40, 80), dtype=np.uint8)
    
    # Create objects with proper sequential IDs
    detected_objects = {
        'player': [
            GameObject('player', (250, 200, 40, 40), 
                      characteristics={'facing_side': 'right'}, 
                      object_id='player_0')
        ],
        'enemy': [
            GameObject('enemy', (100, 150, 30, 30), 
                      characteristics={'facing_side': 'left'}, 
                      object_id='enemy_0'),
            GameObject('enemy', (400, 180, 30, 30), 
                      characteristics={'facing_side': 'right'}, 
                      object_id='enemy_1'),
            GameObject('enemy', (200, 100, 30, 30), 
                      characteristics={'facing_side': 'down'}, 
                      object_id='enemy_2'),
        ],
        'enemy_submarine': [
            GameObject('enemy_submarine', (150, 300, 50, 25), 
                      characteristics={'facing_side': 'left'}, 
                      object_id='enemy_submarine_0'),
            GameObject('enemy_submarine', (350, 320, 50, 25), 
                      characteristics={'facing_side': 'right'}, 
                      object_id='enemy_submarine_1'),
        ],
        'diver': [
            GameObject('diver', (180, 250, 15, 15), object_id='diver_0'),
            GameObject('diver', (320, 240, 15, 15), object_id='diver_1'),
        ],
        'collected_diver': [
            GameObject('collected_diver', (260, 180, 12, 12), object_id='collected_diver_0'),
            GameObject('collected_diver', (270, 180, 12, 12), object_id='collected_diver_1'),
            GameObject('collected_diver', (280, 180, 12, 12), object_id='collected_diver_2'),
            GameObject('collected_diver', (290, 180, 12, 12), object_id='collected_diver_3'),
            GameObject('collected_diver', (300, 180, 12, 12), object_id='collected_diver_4'),
            GameObject('collected_diver', (310, 180, 12, 12), object_id='collected_diver_5'),
        ]
    }
    
    return image, detected_objects

def create_facing_relationships(detected_objects):
    """Create facing side relationships for objects."""
    relationships = []
    
    # Create facing relationships for objects with facing_side characteristic
    for object_type, objects in detected_objects.items():
        for obj in objects:
            if 'facing_side' in obj.characteristics:
                facing_side = obj.characteristics['facing_side']
                # Create a virtual facing side object
                virtual_facing = GameObject('facing_side', (0, 0, 0, 0), object_id=facing_side)
                relationship_type = f'facing{facing_side.capitalize()}'
                relationship = SpatialRelationship(obj, virtual_facing, relationship_type)
                relationships.append(relationship)
    
    return relationships

def test_enhanced_visualization():
    """Test the enhanced visualization features."""
    print("🎨 TESTING ENHANCED VISUALIZATION FEATURES")
    print("=" * 50)
    
    # Create test scene
    image, detected_objects = create_test_scene_with_facing_objects()
    
    # Create facing relationships
    facing_relationships = create_facing_relationships(detected_objects)
    
    # Initialize analyzer to get all relationships (including spatial ones)
    analyzer = SeaquestRelationshipAnalyzer()
    all_relationships = analyzer.analyze_all_relationships(detected_objects)
    
    # Combine facing relationships with spatial relationships
    combined_relationships = facing_relationships + all_relationships
    
    # Create connection list for relationship lines
    connection_list = analyzer.create_connection_list(all_relationships)
    
    # Initialize visualization manager
    visualizer = VisualizationManager()
    visualizer.object_color_mapping = {
        'player': (0, 255, 0),       # Green
        'enemy': (0, 0, 255),        # Red
        'enemy_submarine': (0, 100, 255),  # Orange-ish
        'diver': (255, 0, 0),        # Blue
        'collected_diver': (150, 0, 150),  # Purple
    }
    
    print(f"\n📊 Scene Statistics:")
    for obj_type, objects in detected_objects.items():
        if objects:
            print(f"   {obj_type}: {len(objects)} objects")
    
    print(f"\n🎯 Facing Relationships:")
    for rel in facing_relationships:
        direction = rel.relationship_type.replace('facing', '').lower()
        print(f"   {rel.obj1.object_id}: facing {direction}")
    
    print(f"\n📈 Spatial Relationships: {len(all_relationships)}")
    
    # Create enhanced visualization
    print(f"\n🖼️  Creating enhanced visualization with:")
    print(f"   ✅ Object indexes on top-right corners")
    print(f"   ✅ Directional arrows for facing relationships")
    print(f"   ✅ Colored bounding boxes")
    print(f"   ✅ Relationship lines")
    
    # Use the enhanced comprehensive visualization
    annotated_image = visualizer.create_comprehensive_visualization(
        image=image,
        detected_objects=detected_objects,
        connection_list=connection_list,
        gaze_positions=[],  # No gaze data for this test
        relationships=combined_relationships,  # Pass relationships for arrows
        scale_factor=2
    )
    
    # Add title and legend to the image
    title_text = "Enhanced Visualization: Indexes + Directional Arrows"
    cv2.putText(annotated_image, title_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    legend_y = 60
    legend_items = [
        "Features:",
        "• Numbers in circles = Object indexes",
        "• Yellow arrows = Facing directions",
        "• Green lines = Nearby relationships",
        "• Black lines = Other relationships"
    ]
    
    for item in legend_items:
        cv2.putText(annotated_image, item, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 20
    
    # Save the visualization
    output_filename = 'enhanced_visualization_demo.png'
    cv2.imwrite(output_filename, annotated_image)
    print(f"\n✅ Enhanced visualization saved as: {output_filename}")
    
    # Test individual features
    print(f"\n🔍 Testing Individual Features:")
    
    # Test 1: Just bounding boxes with indexes
    print(f"   Testing bounding boxes with indexes...")
    bbox_image = visualizer.draw_all_objects(image.copy(), detected_objects)
    cv2.imwrite('test_bounding_boxes_with_indexes.png', 
               cv2.resize(bbox_image, (bbox_image.shape[1]*2, bbox_image.shape[0]*2)))
    print(f"   ✅ Saved: test_bounding_boxes_with_indexes.png")
    
    # Test 2: Just directional arrows
    print(f"   Testing directional arrows...")
    arrow_image = image.copy()
    for object_type, objects in detected_objects.items():
        for obj in objects:
            if 'facing_side' in obj.characteristics:
                # Draw basic bounding box
                color = visualizer.object_color_mapping.get(object_type, (255, 255, 255))
                cv2.rectangle(arrow_image, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), color, 2)
                # Draw arrow
                visualizer._draw_arrow_for_direction(arrow_image, obj, obj.characteristics['facing_side'])
    cv2.imwrite('test_directional_arrows.png', 
               cv2.resize(arrow_image, (arrow_image.shape[1]*2, arrow_image.shape[0]*2)))
    print(f"   ✅ Saved: test_directional_arrows.png")
    
    return annotated_image, combined_relationships

def test_object_index_extraction():
    """Test the object index extraction functionality."""
    print(f"\n🔢 TESTING OBJECT INDEX EXTRACTION")
    print("=" * 40)
    
    visualizer = VisualizationManager()
    
    test_cases = [
        ("enemy_0", 0),
        ("enemy_submarine_15", 15),
        ("player_1", 1),
        ("diver_99", 99),
        ("collected_diver_5", 5),
        ("invalid_id", None),
        ("no_underscore", None),
        ("multiple_under_score_3", 3),
    ]
    
    for object_id, expected in test_cases:
        result = visualizer._extract_object_index(object_id)
        status = "✅" if result == expected else "❌"
        result_str = str(result) if result is not None else "None"
        print(f"   {object_id:20} → {result_str:>4} {status}")

if __name__ == "__main__":
    test_enhanced_visualization()
    test_object_index_extraction()
    
    print(f"\n🎉 Enhanced visualization testing complete!")
    print(f"\n📁 Generated files:")
    print(f"   • enhanced_visualization_demo.png (Complete demo)")
    print(f"   • test_bounding_boxes_with_indexes.png (Indexes only)")
    print(f"   • test_directional_arrows.png (Arrows only)")
    print(f"\n💡 Features implemented:")
    print(f"   ✅ Object indexes on top-right corners of bounding boxes")
    print(f"   ✅ Directional arrows for objects with facing relationships")
    print(f"   ✅ Yellow arrows with direction labels (L/R/U/D)")
    print(f"   ✅ Automatic index extraction from object IDs")
