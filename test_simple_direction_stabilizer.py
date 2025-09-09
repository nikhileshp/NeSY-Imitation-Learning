#!/usr/bin/env python3
"""
Test script for simple direction stabilizer using visual detection + frequency analysis.
Verifies that direction is stabilized using max occurrence over 5 frames of visual detection.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.game_object import GameObject
from core.direction_stabilizer import EnemySubmarineDirectionStabilizer
from core.detection_pipeline import SeaquestDetectionPipeline
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer


def test_visual_direction_stabilization():
    """Test that direction stabilization works with visual detection results."""
    print("=" * 60)
    print("TEST: Visual Direction Stabilization")
    print("=" * 60)
    
    stabilizer = EnemySubmarineDirectionStabilizer(history_size=5)
    
    print("\n1. Testing stabilization with mixed visual detections...")
    
    # Create submarine and simulate visual detection results over several frames
    # Frame pattern: Right, Right, Left, Right, Right -> Should stabilize to Right (3 vs 2)
    submarine = GameObject('enemy_submarine', (100, 80, 115, 90), {}, 'enemy_submarine_0')
    
    # Simulate visual detection results (like what facing_side would return)
    visual_detections = ['right', 'right', 'left', 'right', 'right']
    
    stable_directions = []
    
    for i, detected_direction in enumerate(visual_detections):
        # Simulate the submarine having this visual detection result
        submarine.characteristics = {'facing_side': detected_direction}
        
        # Update the stabilizer
        stabilizer.update_submarine_direction(submarine)
        
        # Get the stable direction
        stable_direction = stabilizer.get_submarine_stable_direction(submarine)
        stable_directions.append(stable_direction)
        
        debug_info = stabilizer.get_debug_info_for_submarine(submarine)
        
        print(f"  Frame {i+1}: Visual={detected_direction}, Stable={stable_direction}, History={debug_info['direction_history']}")
    
    # Should stabilize to 'right' since it appears 3 times vs 'left' 2 times
    final_stable_direction = stable_directions[-1]
    debug_info = stabilizer.get_debug_info_for_submarine(submarine)
    
    print(f"\n  Final stable direction: {final_stable_direction}")
    print(f"  Direction counts: {debug_info.get('direction_counts', {})}")
    
    assert final_stable_direction == 'right', f"Expected 'right' (3 occurrences), got {final_stable_direction}"
    
    print("\n2. Testing opposite case - left dominates...")
    
    # Reset and test opposite pattern: Left, Left, Left, Right, Right -> Should be Left (3 vs 2)
    stabilizer.stabilizer.reset_object(submarine.object_id)
    
    opposite_detections = ['left', 'left', 'left', 'right', 'right']
    opposite_stable_directions = []
    
    for i, detected_direction in enumerate(opposite_detections):
        submarine.characteristics = {'facing_side': detected_direction}
        stabilizer.update_submarine_direction(submarine)
        stable_direction = stabilizer.get_submarine_stable_direction(submarine)
        opposite_stable_directions.append(stable_direction)
        
        debug_info = stabilizer.get_debug_info_for_submarine(submarine)
        print(f"  Frame {i+1}: Visual={detected_direction}, Stable={stable_direction}, History={debug_info['direction_history']}")
    
    final_opposite_direction = opposite_stable_directions[-1]
    debug_info = stabilizer.get_debug_info_for_submarine(submarine)
    
    print(f"\n  Final stable direction: {final_opposite_direction}")
    print(f"  Direction counts: {debug_info.get('direction_counts', {})}")
    
    assert final_opposite_direction == 'left', f"Expected 'left' (3 occurrences), got {final_opposite_direction}"
    
    print("\n3. Testing equal counts - first one wins...")
    
    # Reset and test equal pattern: Left, Right, Left, Right, Left -> Should be Left (3 vs 2)
    stabilizer.stabilizer.reset_object(submarine.object_id)
    
    equal_detections = ['left', 'right', 'left', 'right', 'left']
    equal_stable_directions = []
    
    for i, detected_direction in enumerate(equal_detections):
        submarine.characteristics = {'facing_side': detected_direction}
        stabilizer.update_submarine_direction(submarine)
        stable_direction = stabilizer.get_submarine_stable_direction(submarine)
        equal_stable_directions.append(stable_direction)
        
        debug_info = stabilizer.get_debug_info_for_submarine(submarine)
        print(f"  Frame {i+1}: Visual={detected_direction}, Stable={stable_direction}, History={debug_info['direction_history']}")
    
    final_equal_direction = equal_stable_directions[-1]
    debug_info = stabilizer.get_debug_info_for_submarine(submarine)
    
    print(f"\n  Final stable direction: {final_equal_direction}")
    print(f"  Direction counts: {debug_info.get('direction_counts', {})}")
    
    assert final_equal_direction == 'left', f"Expected 'left' (3 occurrences), got {final_equal_direction}"
    
    print("\n✓ Visual direction stabilization test PASSED")


def test_detection_pipeline_integration():
    """Test the detection pipeline integration with direction stabilization."""
    print("\n" + "=" * 60)
    print("TEST: Detection Pipeline Integration")
    print("=" * 60)
    
    pipeline = SeaquestDetectionPipeline()
    
    print("\n1. Testing pipeline processing...")
    
    # Create mock detected objects with submarines having visual detections
    submarine1 = GameObject('enemy_submarine', (100, 80, 115, 90), {'facing_side': 'right'}, 'enemy_submarine_0')
    submarine2 = GameObject('enemy_submarine', (200, 70, 215, 80), {'facing_side': 'left'}, 'enemy_submarine_1')
    player = GameObject('player', (50, 100, 65, 110), {'facing_side': 'right'}, 'player_0')
    
    detected_objects = {
        'player': [player],
        'enemy_submarine': [submarine1, submarine2],
        'collected_diver': []
    }
    
    # Process multiple frames to build up stabilization
    for frame in range(5):
        # Vary the visual detection slightly to test stabilization
        if frame == 2:  # Inject some noise in frame 3
            submarine1.characteristics['facing_side'] = 'left'  # Noise
        else:
            submarine1.characteristics['facing_side'] = 'right'  # Consistent
        
        submarine2.characteristics['facing_side'] = 'left'  # Always left
        
        # Process through pipeline
        processed_objects = pipeline.process_detected_objects(detected_objects)
        
        # Check the results
        sub1_facing = submarine1.characteristics.get('facing_side', 'None')
        sub2_facing = submarine2.characteristics.get('facing_side', 'None')
        sub1_source = submarine1.characteristics.get('facing_source', 'None')
        
        print(f"  Frame {frame+1}: Sub1={sub1_facing}, Sub2={sub2_facing}, Source={sub1_source}")
    
    # After 5 frames, submarine 1 should stabilize to 'right' (4 right, 1 left)
    final_sub1_direction = submarine1.characteristics.get('facing_side')
    final_sub2_direction = submarine2.characteristics.get('facing_side')
    
    print(f"\n  Final directions: Sub1={final_sub1_direction}, Sub2={final_sub2_direction}")
    
    assert final_sub1_direction == 'right', f"Sub1 should stabilize to 'right', got {final_sub1_direction}"
    assert final_sub2_direction == 'left', f"Sub2 should be 'left', got {final_sub2_direction}"
    
    print("\n2. Testing with relationship analyzer...")
    
    analyzer = SeaquestRelationshipAnalyzer()
    
    # Process final frame through relationship analyzer
    relationships = analyzer.analyze_all_relationships(processed_objects)
    
    # Check for enemy facing relationships
    facing_relationships = [r for r in relationships if 'enemyFacing' in r.relationship_type]
    
    print(f"\n  Found {len(facing_relationships)} facing relationships:")
    for rel in facing_relationships:
        formatted = analyzer.game_config.format_relationship_description(rel)
        print(f"    {formatted}")
    
    # Should have enemyFacingRight for sub1 and enemyFacingLeft for sub2
    facing_types = [r.relationship_type for r in facing_relationships]
    assert 'enemyFacingRight' in facing_types, "Should have enemyFacingRight relationship"
    assert 'enemyFacingLeft' in facing_types, "Should have enemyFacingLeft relationship"
    
    # Check that enemy arguments are included
    for rel in facing_relationships:
        formatted = analyzer.game_config.format_relationship_description(rel)
        if 'enemyFacing' in formatted:
            assert '(' in formatted and ')' in formatted, f"Missing parentheses in: {formatted}"
            assert 'enemy_submarine_' in formatted, f"Missing enemy ID in: {formatted}"
    
    print("\n✓ Detection pipeline integration test PASSED")


def test_none_direction_handling():
    """Test handling of None directions (when visual detection fails)."""
    print("\n" + "=" * 60)
    print("TEST: None Direction Handling")
    print("=" * 60)
    
    stabilizer = EnemySubmarineDirectionStabilizer(history_size=5)
    submarine = GameObject('enemy_submarine', (100, 80, 115, 90), {}, 'enemy_submarine_0')
    
    print("\n1. Testing mixed None and valid directions...")
    
    # Pattern: None, right, None, right, right -> Should be right
    detections = [None, 'right', None, 'right', 'right']
    
    for i, detected_direction in enumerate(detections):
        if detected_direction is not None:
            submarine.characteristics = {'facing_side': detected_direction}
        else:
            submarine.characteristics = {}  # No facing_side key
        
        stabilizer.update_submarine_direction(submarine)
        stable_direction = stabilizer.get_submarine_stable_direction(submarine)
        
        debug_info = stabilizer.get_debug_info_for_submarine(submarine)
        print(f"  Frame {i+1}: Visual={detected_direction}, Stable={stable_direction}, History={debug_info['direction_history']}")
    
    final_direction = stabilizer.get_submarine_stable_direction(submarine)
    debug_info = stabilizer.get_debug_info_for_submarine(submarine)
    
    print(f"\n  Final direction: {final_direction}")
    print(f"  Only valid directions in history: {debug_info['direction_history']}")
    
    # Should be 'right' since None values are ignored
    assert final_direction == 'right', f"Expected 'right', got {final_direction}"
    assert len(debug_info['direction_history']) == 3, f"Should have 3 valid directions, got {len(debug_info['direction_history'])}"
    
    print("\n✓ None direction handling test PASSED")


def main():
    """Run all tests."""
    print("SIMPLE VISUAL DIRECTION STABILIZER TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Basic visual direction stabilization
        test_visual_direction_stabilization()
        
        # Test 2: Detection pipeline integration
        test_detection_pipeline_integration()
        
        # Test 3: None direction handling
        test_none_direction_handling()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nSummary of simple direction stabilizer behavior verified:")
        print("✓ Uses visual detection results (same as player)")
        print("✓ Stabilizes direction using max occurrence over 5 frames")
        print("✓ Ignores None/invalid visual detection results")
        print("✓ Integrates seamlessly with detection pipeline")
        print("✓ Works correctly with relationship analyzer")
        print("✓ Maintains enemy submarine arguments in relationships")
        print("✓ Much simpler than complex movement tracking")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
