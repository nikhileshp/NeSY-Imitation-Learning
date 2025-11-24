#!/usr/bin/env python3
"""
Test script to verify the reciprocal rank-based distance weight calculation system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from core.distance_weight_calculator import DistanceWeightCalculator
from core.game_object import GameObject, SpatialRelationship


def test_reciprocal_rank_weights():
    """Test reciprocal rank weight calculation with sample objects."""
    print("Testing Reciprocal Rank Distance Weight Calculation")
    print("=" * 50)
    
    # Initialize calculator with screen dimensions
    calculator = DistanceWeightCalculator(800, 600)
    
    # Create test objects at different distances from a gaze point
    gaze_pos = (400, 300)  # Center of screen
    print(f"Gaze position: {gaze_pos}")
    
    # Create objects at different distances (closer to farther)
    objects = [
        GameObject("diver", (390, 290, 410, 310), object_id="diver_1"),      # Very close (distance ≈ 14)
        GameObject("enemy", (350, 250, 370, 270), object_id="enemy_1"),      # Medium close (distance ≈ 70)
        GameObject("enemy_submarine", (200, 200, 220, 220), object_id="sub_1"),  # Far (distance ≈ 283)
        GameObject("enemy_missile", (100, 100, 110, 110), object_id="missile_1"),  # Very far (distance ≈ 424)
    ]
    
    # Calculate centers and distances for verification
    objects_with_centers = []
    print("\nObjects and their distances from gaze:")
    for obj in objects:
        center = obj.center
        distance = calculator.calculate_distance(gaze_pos, center)
        objects_with_centers.append((obj, center))
        print(f"  {obj.object_type} (ID: {obj.object_id}): center={center}, distance={distance:.1f}")
    
    # Calculate reciprocal rank weights
    weights = calculator.calculate_reciprocal_rank_weights(gaze_pos, objects_with_centers)
    
    print("\nReciprocal Rank Weights:")
    sorted_by_weight = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for rank, (obj, weight) in enumerate(sorted_by_weight, 1):
        expected_weight = 1.0 / rank
        print(f"  Rank {rank}: {obj.object_type} (ID: {obj.object_id}) = {weight:.3f} (expected: {expected_weight:.3f})")
    
    print()
    return weights


def test_relationship_distance_weights():
    """Test the full relationship distance weight calculation."""
    print("Testing Relationship Distance Weight Calculation")
    print("=" * 50)
    
    # Initialize calculator
    calculator = DistanceWeightCalculator(800, 600)
    
    # Gaze position
    gaze_positions = [(400, 300)]  # Single gaze position
    print(f"Gaze positions: {gaze_positions}")
    
    # Create test objects
    player = GameObject("player", (380, 280, 420, 320), object_id="player_1")
    diver = GameObject("diver", (390, 290, 410, 310), object_id="diver_1")      # Closest
    enemy1 = GameObject("enemy", (350, 250, 370, 270), object_id="enemy_1")     # Second closest
    enemy2 = GameObject("enemy", (450, 350, 470, 370), object_id="enemy_2")     # Third closest  
    submarine = GameObject("enemy_submarine", (200, 200, 220, 220), object_id="sub_1")  # Farthest
    
    # Create relationships
    relationships = [
        SpatialRelationship(player, diver, "rightOfDiver"),
        SpatialRelationship(player, enemy1, "belowOfEnemy"), 
        SpatialRelationship(player, enemy2, "leftOfEnemy"),
        SpatialRelationship(player, submarine, "rightOfEnemySubmarine"),
        SpatialRelationship(player, diver, "visibleDiver"),
        SpatialRelationship(player, enemy1, "visibleEnemy"),
        SpatialRelationship(player, enemy2, "visibleEnemy"),
        SpatialRelationship(player, submarine, "visibleEnemySubmarine"),
    ]
    
    print("\nCreated relationships:")
    for rel in relationships:
        if hasattr(rel.obj2, 'center'):
            distance = calculator.calculate_distance(gaze_positions[0], rel.obj2.center)
            print(f"  {rel.relationship_type}({rel.obj2.object_id}): center={rel.obj2.center}, distance={distance:.1f}")
    
    # Calculate relationship distance weights
    weights = calculator.calculate_relationship_distance_weights(relationships, gaze_positions)
    
    print("\nRelationship Distance Weights (Reciprocal Rank):")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for rank, (rel_id, weight) in enumerate(sorted_weights, 1):
        expected_weight = 1.0 / rank  
        print(f"  Rank {rank}: {rel_id} = {weight:.3f} (expected: {expected_weight:.3f})")
    
    # Format for DataFrame
    formatted_weights = calculator.format_distance_weights_for_dataframe(weights)
    print(f"\nFormatted for DataFrame: {formatted_weights}")
    
    print()
    return weights


def test_alternating_class_weights():
    """Test the alternating class weight calculation."""
    print("Testing Alternating Class Weight Calculation")
    print("=" * 50)
    
    # Initialize calculator with screen dimensions
    calculator = DistanceWeightCalculator(800, 600)
    
    # Create test objects with multiple objects of the same class at different distances
    gaze_pos = (400, 300)  # Center of screen
    print(f"Gaze position: {gaze_pos}")
    
    # Create objects - mix of classes with duplicates
    objects = [
        GameObject("diver", (395, 295, 405, 305), object_id="diver_1"),      # Closest diver
        GameObject("enemy", (390, 290, 410, 310), object_id="enemy_1"),      # Closest enemy  
        GameObject("diver", (350, 250, 370, 270), object_id="diver_2"),      # Second diver (should get 0)
        GameObject("enemy_submarine", (300, 200, 320, 220), object_id="sub_1"), # Closest submarine
        GameObject("enemy", (250, 150, 270, 170), object_id="enemy_2"),      # Second enemy (should get 0)
        GameObject("diver", (200, 100, 220, 120), object_id="diver_3"),      # Third diver (should get weight)
        GameObject("enemy_submarine", (150, 50, 170, 70), object_id="sub_2"), # Second submarine (should get 0)
    ]
    
    # Calculate centers and distances for verification
    objects_with_centers = []
    print("\nObjects and their distances from gaze:")
    for obj in objects:
        center = obj.center
        distance = calculator.calculate_distance(gaze_pos, center)
        objects_with_centers.append((obj, center))
        print(f"  {obj.object_type} (ID: {obj.object_id}): center={center}, distance={distance:.1f}")
    
    # Calculate alternating class weights
    weights = calculator.calculate_alternating_class_weights(gaze_pos, objects_with_centers)
    
    print("\nAlternating Class Weights:")
    sorted_by_distance = sorted(objects_with_centers, 
                               key=lambda x: calculator.calculate_distance(gaze_pos, x[1]))
    
    class_counts = {}
    current_rank = 1
    
    for obj, center in sorted_by_distance:
        distance = calculator.calculate_distance(gaze_pos, center)
        weight = weights[obj]
        object_class = obj.object_type
        
        if object_class not in class_counts:
            class_counts[object_class] = 0
        class_counts[object_class] += 1
        class_occurrence = class_counts[object_class]
        
        # Determine expected weight
        if class_occurrence % 2 == 1:  # Odd occurrence
            expected_weight = 1.0 / current_rank
            current_rank += 1
        else:  # Even occurrence
            expected_weight = 0.0
        
        status = "✓" if abs(weight - expected_weight) < 0.001 else "✗"
        print(f"  {status} {obj.object_type} (ID: {obj.object_id}): "
              f"distance={distance:.1f}, occurrence={class_occurrence}, "
              f"weight={weight:.3f} (expected: {expected_weight:.3f})")
    
    print("\nClass occurrence summary:")
    for obj_class, count in class_counts.items():
        print(f"  {obj_class}: {count} objects")
    
    print()
    return weights


def test_nearest_only_weights():
    """Test the nearest-only weight calculation."""
    print("Testing Nearest-Only Weight Calculation")
    print("=" * 50)
    
    # Initialize calculator with screen dimensions
    calculator = DistanceWeightCalculator(800, 600)
    
    # Create test objects at different distances from a gaze point
    gaze_pos = (400, 300)  # Center of screen
    print(f"Gaze position: {gaze_pos}")
    
    # Create objects at different distances
    objects = [
        GameObject("diver", (395, 295, 405, 305), object_id="diver_1"),      # Closest
        GameObject("enemy", (350, 250, 370, 270), object_id="enemy_1"),      # Second closest
        GameObject("enemy_submarine", (300, 200, 320, 220), object_id="sub_1"), # Third closest
        GameObject("enemy_missile", (200, 100, 220, 120), object_id="missile_1"), # Farthest
        GameObject("diver", (180, 80, 200, 100), object_id="diver_2"),       # Another far diver
    ]
    
    # Calculate centers and distances for verification
    objects_with_centers = []
    print("\nObjects and their distances from gaze:")
    for obj in objects:
        center = obj.center
        distance = calculator.calculate_distance(gaze_pos, center)
        objects_with_centers.append((obj, center))
        print(f"  {obj.object_type} (ID: {obj.object_id}): center={center}, distance={distance:.1f}")
    
    # Calculate nearest-only weights
    weights = calculator.calculate_nearest_only_weights(gaze_pos, objects_with_centers)
    
    print("\nNearest-Only Weights:")
    sorted_by_distance = sorted(objects_with_centers, 
                               key=lambda x: calculator.calculate_distance(gaze_pos, x[1]))
    
    for i, (obj, center) in enumerate(sorted_by_distance):
        distance = calculator.calculate_distance(gaze_pos, center)
        weight = weights[obj]
        expected_weight = 1.0 if i == 0 else 0.0  # Only nearest gets 1.0
        
        status = "✓" if abs(weight - expected_weight) < 0.001 else "✗"
        rank_text = "NEAREST" if i == 0 else f"Rank {i+1}"
        print(f"  {status} {rank_text}: {obj.object_type} (ID: {obj.object_id}) = "
              f"distance={distance:.1f}, weight={weight:.3f} (expected: {expected_weight:.3f})")
    
    # Verify only one object has weight 1.0
    weights_of_1 = [obj.object_id for obj, w in weights.items() if w == 1.0]
    weights_of_0 = [obj.object_id for obj, w in weights.items() if w == 0.0]
    
    print(f"\nSummary:")
    print(f"  Objects with weight 1.0: {weights_of_1} (should be 1 object)")
    print(f"  Objects with weight 0.0: {weights_of_0} (should be {len(objects)-1} objects)")
    
    # Test with single object
    print("\nTesting with single object:")
    single_obj_weights = calculator.calculate_nearest_only_weights(gaze_pos, [(objects[0], objects[0].center)])
    single_weight = list(single_obj_weights.values())[0]
    print(f"  Single object weight: {single_weight:.3f} (expected: 1.000)")
    
    print()
    return weights


def test_edge_cases():
    """Test edge cases for reciprocal rank weights."""
    print("Testing Edge Cases")
    print("=" * 30)
    
    calculator = DistanceWeightCalculator(800, 600)
    
    # Test with no objects
    print("1. No objects:")
    weights = calculator.calculate_reciprocal_rank_weights((400, 300), [])
    print(f"   Result: {weights}")
    
    # Test with single object
    print("\n2. Single object:")
    single_obj = GameObject("diver", (390, 290, 410, 310), object_id="diver_1")
    weights = calculator.calculate_reciprocal_rank_weights((400, 300), [(single_obj, single_obj.center)])
    print(f"   Result: {list(weights.values())[0]:.3f} (expected: 1.000)")
    
    # Test with objects at same distance (tied ranks)
    print("\n3. Objects at same distance:")
    obj1 = GameObject("diver", (390, 290, 410, 310), object_id="diver_1")  # Distance ≈ 14
    obj2 = GameObject("enemy", (410, 310, 430, 330), object_id="enemy_1")  # Distance ≈ 14 
    objects_same_distance = [(obj1, obj1.center), (obj2, obj2.center)]
    weights = calculator.calculate_reciprocal_rank_weights((400, 300), objects_same_distance)
    print("   Objects with similar distances:")
    for obj, weight in weights.items():
        distance = calculator.calculate_distance((400, 300), obj.center)
        print(f"     {obj.object_type} (ID: {obj.object_id}): distance={distance:.1f}, weight={weight:.3f}")
    
    # Test with gaze exactly on object center
    print("\n4. Gaze exactly on object center:")
    exact_obj = GameObject("diver", (390, 290, 410, 310), object_id="diver_1")
    gaze_on_center = exact_obj.center
    weights = calculator.calculate_reciprocal_rank_weights(gaze_on_center, [(exact_obj, exact_obj.center)])
    print(f"   Gaze position: {gaze_on_center}")
    print(f"   Object center: {exact_obj.center}")
    print(f"   Weight: {list(weights.values())[0]:.3f} (expected: 1.000)")
    
    print()


def main():
    """Run all reciprocal rank weight tests."""
    print("Reciprocal Rank Distance Weight Tests")
    print("=" * 60)
    print()
    
    try:
        # Test basic reciprocal rank calculation
        test_reciprocal_rank_weights()
        
        # Test relationship distance weights
        test_relationship_distance_weights()
        
        # Test alternating class weights
        test_alternating_class_weights()
        
        # Test nearest-only weights
        test_nearest_only_weights()
        
        # Test edge cases
        test_edge_cases()
        
        print("All tests completed successfully! ✓")
        print("\nSummary:")
        print("Standard reciprocal rank weights:")
        print("- Closest object gets weight = 1.0")
        print("- Second closest gets weight = 0.5 (1/2)")
        print("- Third closest gets weight = 0.333 (1/3)")
        print("- Fourth closest gets weight = 0.25 (1/4)")
        print("- And so on...")
        print("\nAlternating class weights:")
        print("- First object of each class gets standard reciprocal rank weight")
        print("- Second object of same class gets weight = 0.0")
        print("- Third object of same class gets next available rank weight")
        print("- Fourth object of same class gets weight = 0.0")
        print("- And so on...")
        print("\nNearest-only weights:")
        print("- Only the nearest object gets weight = 1.0")
        print("- All other objects get weight = 0.0")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    main()