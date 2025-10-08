#!/usr/bin/env python3
"""
Example script demonstrating distance weight calculation functionality.
This shows how distance weights are calculated for relationships between gaze coordinates
and spatial objects like divers, enemies, and enemy submarines.
"""
import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.distance_weight_calculator import DistanceWeightCalculator
from core.game_object import GameObject, SpatialRelationship


def demonstrate_distance_weights():
    """Demonstrate distance weight calculation with a realistic game scenario."""
    print("🎮 Distance Weight Calculator Example")
    print("=" * 60)
    
    # Create calculator for typical Seaquest game screen (160x210 scaled up to 640x840)
    screen_width, screen_height = 640, 840
    calculator = DistanceWeightCalculator(screen_width, screen_height)
    
    print(f"Screen dimensions: {screen_width} x {screen_height}")
    print(f"Maximum possible distance: {calculator.max_possible_distance:.2f} pixels")
    print()
    
    # Create game objects representing a typical Seaquest frame
    print("🐟 Creating game objects:")
    player = GameObject("player", (320, 400, 20, 30), object_id="player_1")
    print(f"  Player at center: {player.center}")
    
    # Divers at various positions
    diver1 = GameObject("diver", (100, 350, 15, 20), object_id="diver_1")
    diver2 = GameObject("diver", (500, 300, 15, 20), object_id="diver_2")
    print(f"  Diver 1 at: {diver1.center}")
    print(f"  Diver 2 at: {diver2.center}")
    
    # Enemy submarine and regular enemies
    enemy_sub = GameObject("enemy_submarine", (200, 500, 40, 20), object_id="sub_1")
    enemy1 = GameObject("enemy", (450, 200, 25, 25), object_id="enemy_1")
    enemy2 = GameObject("enemy", (150, 600, 25, 25), object_id="enemy_2")
    print(f"  Enemy submarine at: {enemy_sub.center}")
    print(f"  Enemy 1 at: {enemy1.center}")
    print(f"  Enemy 2 at: {enemy2.center}")
    print()
    
    # Create relationships that involve spatial objects
    print("🔗 Creating relationships:")
    relationships = [
        SpatialRelationship(player, diver1, "nearbyDiver"),
        SpatialRelationship(player, diver2, "rightOfDiver"), 
        SpatialRelationship(player, enemy1, "belowOfEnemy"),
        SpatialRelationship(player, enemy2, "rightOfEnemy"),
        SpatialRelationship(diver1, GameObject("visibility_state", (0, 0, 0, 0), object_id="visible"), "visibleDiver"),
        SpatialRelationship(diver2, GameObject("visibility_state", (0, 0, 0, 0), object_id="visible"), "visibleDiver"),
        SpatialRelationship(enemy_sub, GameObject("visibility_state", (0, 0, 0, 0), object_id="visible"), "visibleEnemySubmarine"),
        SpatialRelationship(enemy1, GameObject("visibility_state", (0, 0, 0, 0), object_id="visible"), "visibleEnemy"),
        SpatialRelationship(enemy2, GameObject("visibility_state", (0, 0, 0, 0), object_id="visible"), "visibleEnemy")
    ]
    
    for rel in relationships:
        print(f"  {rel.relationship_type}: {rel.obj1.object_id} -> {rel.obj2.object_id}")
    print()
    
    # Simulate gaze positions - player looking at different objects over time
    print("👁️  Simulating gaze positions (using last position as displayed gaze):")
    gaze_scenarios = [
        {
            "name": "Looking at Diver 1",
            "positions": [(105, 360), (95, 355), (110, 365)],  # Last: close to diver1
            "displayed": "(110, 365) - close to Diver 1"
        },
        {
            "name": "Looking at Enemy Submarine", 
            "positions": [(220, 510), (215, 505), (225, 515)],  # Last: close to enemy_sub
            "displayed": "(225, 515) - close to Enemy Submarine"
        },
        {
            "name": "Looking at Enemy 1",
            "positions": [(460, 210), (455, 205), (465, 215)],  # Last: close to enemy1
            "displayed": "(465, 215) - close to Enemy 1"
        },
        {
            "name": "Looking away from all objects",
            "positions": [(50, 50), (600, 800), (320, 750)],  # Last: center-bottom
            "displayed": "(320, 750) - center-bottom, away from objects"
        }
    ]
    
    # Calculate and display weights for each scenario
    for scenario in gaze_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   All gaze positions: {scenario['positions']}")
        print(f"   Displayed gaze position: {scenario['displayed']}")
        
        # Calculate distance weights
        distance_weights = calculator.calculate_relationship_distance_weights(
            relationships, scenario['positions']
        )
        
        if distance_weights:
            print("   Individual relationship distance weights:")
            for rel_identifier in sorted(distance_weights.keys()):
                print(f"     {rel_identifier:30}: {distance_weights[rel_identifier]:6.2f}")
            
            # Show formatted output for DataFrame storage
            formatted = calculator.format_distance_weights_for_dataframe(distance_weights)
            print(f"   DataFrame format: {formatted}")
        else:
            print("   No distance weights calculated (no spatial object relationships)")
    
    print("\n" + "=" * 60)
    print("✅ Distance weight calculation example completed!")
    print("\nKey insights:")
    print("• Higher weights indicate closer gaze-to-object distances")
    print("• Weights are calculated as max_possible_distance / actual_distance") 
    print("• Only relationships involving divers, enemies, enemy_submarines are weighted")
    print("• Both max and average weights are stored for analysis flexibility")


if __name__ == "__main__":
    demonstrate_distance_weights()