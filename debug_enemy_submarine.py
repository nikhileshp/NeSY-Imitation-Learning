#!/usr/bin/env python3
"""
Debug script to specifically verify enemy submarine facing side relationships and visualization.
"""

import numpy as np
import cv2
from core.game_object import GameObject
from core.object_tracker import TrackableGameObject
from core.visualization_manager import VisualizationManager
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer


def test_enemy_submarine_specific():
    """Test enemy submarine specific behavior."""
    print("🔍 DEBUG: ENEMY SUBMARINE FACING RELATIONSHIPS")
    print("=" * 55)
    
    # Create test scene with only enemy submarines
    image = np.full((300, 400, 3), (40, 40, 80), dtype=np.uint8)
    
    detected_objects = {
        'enemy_submarine': [
            TrackableGameObject('enemy_submarine', (100, 150, 50, 25), 
                              characteristics={'facing_side': 'left'}, object_id='enemy_submarine_0'),
            TrackableGameObject('enemy_submarine', (250, 200, 50, 25), 
                              characteristics={'facing_side': 'right'}, object_id='enemy_submarine_1'),
        ]
    }
    
    print("📋 Created enemy submarines:")
    for obj in detected_objects['enemy_submarine']:
        print(f"   {obj.object_id}: {obj.characteristics}")
    print()
    
    # Test relationship analysis
    analyzer = SeaquestRelationshipAnalyzer()
    relationships = analyzer.analyze_all_relationships(detected_objects)
    
    print("🔗 Generated relationships:")
    enemy_facing_relationships = []
    for rel in relationships:
        if rel.relationship_type.startswith('enemyFacing'):
            enemy_facing_relationships.append(rel)
            print(f"   ✅ {rel.obj1.object_id} -> {rel.relationship_type}")
    
    if not enemy_facing_relationships:
        print("   ❌ NO ENEMY FACING RELATIONSHIPS FOUND!")
    print()
    
    # Test visualization
    visualizer = VisualizationManager()
    visualizer.object_color_mapping = {
        'enemy_submarine': (0, 100, 255),  # Orange-ish color
    }
    
    print("🎨 Testing visualization:")
    
    # Test with relationships
    vis_with_relationships = visualizer.draw_all_objects(image.copy(), detected_objects, relationships)
    cv2.putText(vis_with_relationships, "Enemy Submarines with Relationships", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Test with characteristics only
    vis_with_characteristics = visualizer.draw_all_objects(image.copy(), detected_objects)
    cv2.putText(vis_with_characteristics, "Enemy Submarines with Characteristics Only", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save debug images
    cv2.imwrite('debug_enemy_submarine_relationships.png', 
                cv2.resize(vis_with_relationships, (vis_with_relationships.shape[1]*3, vis_with_relationships.shape[0]*3)))
    cv2.imwrite('debug_enemy_submarine_characteristics.png', 
                cv2.resize(vis_with_characteristics, (vis_with_characteristics.shape[1]*3, vis_with_characteristics.shape[0]*3)))
    
    print("   ✅ Saved debug_enemy_submarine_relationships.png")
    print("   ✅ Saved debug_enemy_submarine_characteristics.png")
    print()
    
    # Check visualization details
    print("🔧 Visualization details:")
    print(f"   • Number of enemy submarines: {len(detected_objects['enemy_submarine'])}")
    print(f"   • Number of enemyFacing relationships: {len(enemy_facing_relationships)}")
    print(f"   • Images saved at 3x scale for better visibility")
    
    if len(enemy_facing_relationships) == 2:
        print("   ✅ SUCCESS: Both enemy submarines have enemyFacing relationships!")
        print("   ✅ SUCCESS: Arrows should be visible in both visualization modes!")
    else:
        print("   ❌ ISSUE: Missing enemy facing relationships!")
    
    return relationships


if __name__ == "__main__":
    test_enemy_submarine_specific()
    print()
    print("🎯 RESULT: Check the generated debug images to verify:")
    print("   1. Enemy submarines have yellow arrows pointing in correct directions")
    print("   2. Both relationship-based and characteristic-based visualization work")
    print("   3. Object indexes (0, 1) are displayed in top-right corners")
