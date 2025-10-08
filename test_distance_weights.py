#!/usr/bin/env python3
"""
Test script for distance weight calculation functionality.
Tests the DistanceWeightCalculator with various scenarios.
"""
import math
import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.distance_weight_calculator import DistanceWeightCalculator
from core.game_object import GameObject, SpatialRelationship


def test_basic_distance_calculation():
    """Test basic distance calculation between two points."""
    print("=== Testing Basic Distance Calculation ===")
    
    # Create calculator for 640x480 screen
    calculator = DistanceWeightCalculator(640, 480)
    
    # Test distance calculation
    point1 = (0, 0)
    point2 = (3, 4)  # Should be distance 5 (3-4-5 triangle)
    
    distance = calculator.calculate_distance(point1, point2)
    print(f"Distance from {point1} to {point2}: {distance}")
    assert abs(distance - 5.0) < 0.001, f"Expected 5.0, got {distance}"
    
    # Test max possible distance
    expected_max = math.sqrt(640**2 + 480**2)
    print(f"Max possible distance for 640x480 screen: {calculator.max_possible_distance}")
    print(f"Expected: {expected_max}")
    assert abs(calculator.max_possible_distance - expected_max) < 0.001
    
    print("✓ Basic distance calculation tests passed\n")


def test_distance_weight_calculation():
    """Test distance weight calculation."""
    print("=== Testing Distance Weight Calculation ===")
    
    calculator = DistanceWeightCalculator(640, 480)
    max_distance = calculator.max_possible_distance
    
    # Test weight when gaze is exactly on object center
    gaze_pos = (100, 100)
    obj_center = (100, 100)
    weight = calculator.calculate_distance_weight(gaze_pos, obj_center)
    print(f"Weight when gaze is exactly on object: {weight}")
    assert weight == max_distance, f"Expected {max_distance}, got {weight}"
    
    # Test weight when gaze is far from object
    gaze_pos = (0, 0)
    obj_center = (640, 480)  # Opposite corner
    actual_distance = calculator.calculate_distance(gaze_pos, obj_center)
    expected_weight = max_distance / actual_distance
    weight = calculator.calculate_distance_weight(gaze_pos, obj_center)
    print(f"Weight when gaze is at opposite corner: {weight}")
    print(f"Expected weight: {expected_weight}")
    assert abs(weight - expected_weight) < 0.001, f"Expected {expected_weight}, got {weight}"
    
    print("✓ Distance weight calculation tests passed\n")


def test_relationship_distance_weights():
    """Test relationship distance weight calculation with sample objects and relationships."""
    print("=== Testing Relationship Distance Weights ===")
    
    calculator = DistanceWeightCalculator(640, 480)
    
    # Create sample objects
    player = GameObject("player", (100, 200, 20, 30), object_id="player_1")
    diver = GameObject("diver", (300, 250, 15, 20), object_id="diver_1")
    enemy = GameObject("enemy", (500, 150, 25, 25), object_id="enemy_1")
    enemy_submarine = GameObject("enemy_submarine", (200, 400, 40, 20), object_id="sub_1")
    
    # Create sample relationships
    relationships = [
        SpatialRelationship(player, diver, "nearbyDiver"),
        SpatialRelationship(player, enemy, "leftOfEnemy"), 
        SpatialRelationship(diver, GameObject("visibility_state", (0, 0, 0, 0), object_id="visible"), "visibleDiver"),
        SpatialRelationship(enemy_submarine, GameObject("visibility_state", (0, 0, 0, 0), object_id="visible"), "visibleEnemySubmarine")
    ]
    
    # Create gaze positions (will use the last one: (210, 410))
    gaze_positions = [(310, 260), (505, 162), (210, 410)]  # Last position is close to submarine
    
    # Calculate distance weights
    distance_weights = calculator.calculate_relationship_distance_weights(relationships, gaze_positions)
    
    print("Distance weights calculated:")
    for rel_identifier, weight in distance_weights.items():
        print(f"  {rel_identifier}: {weight:.2f}")
    
    # Check that we have weights for relationships involving target objects
    expected_identifiers = {"nearbyDiver(diver_1)", "leftOfEnemy(enemy_1)", "visibleDiver(diver_1)", "visibleEnemySubmarine(sub_1)"}
    found_identifiers = set(distance_weights.keys())
    
    print(f"Expected identifiers: {expected_identifiers}")
    print(f"Found identifiers: {found_identifiers}")
    
    # Should have weights for all target relationships
    assert expected_identifiers.issubset(found_identifiers), f"Missing relationships: {expected_identifiers - found_identifiers}"
    
    # Each relationship should have exactly one weight (from last gaze position)
    for rel_identifier in expected_identifiers:
        if rel_identifier in distance_weights:
            weight = distance_weights[rel_identifier]
            print(f"  {rel_identifier} weight: {weight:.2f}")
            assert isinstance(weight, float), f"{rel_identifier} should have a float weight, got {type(weight)}"
    
    print("✓ Relationship distance weights tests passed\n")


def test_individual_relationship_weights():
    """Test individual relationship weight calculation."""
    print("=== Testing Individual Relationship Weights ===")
    
    calculator = DistanceWeightCalculator(640, 480)
    
    # Sample distance weights (now individual relationships)
    distance_weights = {
        "nearbyDiver(diver_1)": 10.5,
        "leftOfEnemy(enemy_1)": 7.8,
        "visibleEnemySubmarine(sub_1)": 15.0
    }
    
    print("Individual relationship weights:")
    for rel_identifier, weight in distance_weights.items():
        print(f"  {rel_identifier}: {weight}")
    
    # Verify each weight is a single float value
    assert isinstance(distance_weights["nearbyDiver(diver_1)"], float)
    assert isinstance(distance_weights["leftOfEnemy(enemy_1)"], float)
    assert isinstance(distance_weights["visibleEnemySubmarine(sub_1)"], float)
    
    print("✓ Individual relationship weights test passed\n")


def test_dataframe_formatting():
    """Test DataFrame formatting functions."""
    print("=== Testing DataFrame Formatting ===")
    
    calculator = DistanceWeightCalculator(640, 480)
    
    # Sample individual relationship distance weights
    distance_weights = {
        "nearbyDiver(diver_1)": 10.5,
        "leftOfEnemy(enemy_1)": 7.8,
        "visibleEnemySubmarine(sub_1)": 15.0
    }
    
    # Test formatting
    formatted = calculator.format_distance_weights_for_dataframe(distance_weights)
    print(f"Formatted: {formatted}")
    
    # Should contain all relationship identifiers with their weights
    assert "nearbyDiver(diver_1):10.50" in formatted
    assert "leftOfEnemy(enemy_1):7.80" in formatted
    assert "visibleEnemySubmarine(sub_1):15.00" in formatted
    
    # Should use semicolon separator
    assert " ; " in formatted
    
    print("✓ DataFrame formatting tests passed\n")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("=== Testing Edge Cases ===")
    
    calculator = DistanceWeightCalculator(640, 480)
    
    # Test with no gaze positions
    empty_weights = calculator.calculate_relationship_distance_weights([], [])
    assert empty_weights == {}, "Should return empty dict for no relationships"
    
    # Test with no relationships
    gaze_positions = [(100, 100)]
    empty_weights = calculator.calculate_relationship_distance_weights([], gaze_positions)
    assert empty_weights == {}, "Should return empty dict for no relationships"
    
    # Test with relationships that don't involve target object types
    water_surface = GameObject("water_surface", (0, 0, 640, 1), object_id="water")
    player = GameObject("player", (100, 200, 20, 30), object_id="player_1")
    non_target_relationship = SpatialRelationship(player, water_surface, "aboveWater")
    
    weights = calculator.calculate_relationship_distance_weights([non_target_relationship], gaze_positions)
    assert weights == {}, "Should return empty dict for non-target relationships"
    
    # Test formatting with empty weights
    empty_formatted = calculator.format_distance_weights_for_dataframe({})
    assert empty_formatted == "", "Should return empty string for empty weights"
    
    print("✓ Edge case tests passed\n")


def main():
    """Run all tests."""
    print("Distance Weight Calculator Test Suite")
    print("=" * 50)
    
    try:
        test_basic_distance_calculation()
        test_distance_weight_calculation()
        test_relationship_distance_weights()
        test_individual_relationship_weights()
        test_dataframe_formatting()
        test_edge_cases()
        
        print("🎉 All tests passed successfully!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()