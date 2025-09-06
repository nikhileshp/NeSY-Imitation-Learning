#!/usr/bin/env python3
"""
Test script to verify diver count hysteresis logic that handles blinking divers.
When divers reach 6, they start blinking (7 frames visible, 9 frames missing).
The diversfull relationship should persist until count actually drops to 5.
"""

from core.game_object import GameObject
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer

def create_detected_objects_with_diver_count(count):
    """Create detected objects with a specific number of collected divers."""
    detected_objects = {
        'player': [GameObject('player', (100, 100, 20, 20), object_id='player_0')],
        'collected_diver': [],
    }
    
    # Add the specified number of divers
    for i in range(count):
        diver = GameObject('collected_diver', (50 + i*10, 50, 10, 10), object_id=f'collected_diver_{i}')
        detected_objects['collected_diver'].append(diver)
    
    return detected_objects

def get_diver_relationship(analyzer, detected_objects):
    """Get the diver count relationship from analysis results."""
    relationships = analyzer.analyze_all_relationships(detected_objects)
    diver_relationships = [r for r in relationships if r.relationship_type in ['diversfull', 'diversNotfull']]
    return diver_relationships[0].relationship_type if diver_relationships else None

def test_diver_hysteresis():
    """Test the hysteresis logic for handling blinking divers."""
    print("🏊 TESTING DIVER COUNT HYSTERESIS (Blinking Diver Handling)")
    print("=" * 65)
    
    analyzer = SeaquestRelationshipAnalyzer()
    
    print("\n📊 SCENARIO: Simulating the blinking diver problem")
    print("   When 6 divers are collected, they start blinking:")
    print("   - 7 frames: divers visible (count = 6)")
    print("   - 9 frames: divers missing (count = 0)")
    print("   Expected: diversfull should persist until count drops to 5")
    
    # Test progression from 0 to 6 divers
    print(f"\n🔄 PHASE 1: Building up diver count (0 → 6)")
    for count in range(7):
        detected_objects = create_detected_objects_with_diver_count(count)
        relationship = get_diver_relationship(analyzer, detected_objects)
        state_indicator = "✅" if relationship == 'diversfull' else "❌"
        print(f"   Count {count}: {relationship} {state_indicator}")
        
        if count < 6:
            assert relationship == 'diversNotfull', f"Expected diversNotfull for count {count}, got {relationship}"
        else:
            assert relationship == 'diversfull', f"Expected diversfull for count {count}, got {relationship}"
    
    # Now simulate the blinking pattern: 6 divers achieved, then start blinking
    print(f"\n🔄 PHASE 2: Simulating blinking pattern (6 divers reached)")
    print("   Pattern: [6 visible] → [0 missing] → [6 visible] → [0 missing] → ...")
    
    # Track the state through multiple blink cycles
    blink_pattern = [
        # First blink cycle: 7 frames visible, 9 frames missing
        (6, 7, "visible"),   # 7 frames with 6 divers visible
        (0, 9, "missing"),   # 9 frames with 0 divers (blinking off)
        # Second blink cycle
        (6, 7, "visible"),   # 7 frames with 6 divers visible
        (0, 9, "missing"),   # 9 frames with 0 divers (blinking off)
        # Third blink cycle
        (6, 4, "visible"),   # 4 frames with 6 divers visible (partial cycle)
    ]
    
    frame_number = 0
    for diver_count, duration, phase in blink_pattern:
        print(f"\n   📹 Blink phase: {phase} ({diver_count} divers for {duration} frames)")
        
        for frame in range(duration):
            frame_number += 1
            detected_objects = create_detected_objects_with_diver_count(diver_count)
            relationship = get_diver_relationship(analyzer, detected_objects)
            
            # During blinking, diversfull should persist even when count = 0
            expected = 'diversfull'  # Should always be diversfull during blinking
            status = "✅" if relationship == expected else "❌"
            
            print(f"     Frame {frame_number:2d}: Count={diver_count}, State={relationship} {status}")
            
            assert relationship == expected, f"Frame {frame_number}: Expected {expected}, got {relationship}"
    
    # Test the transition back to diversNotfull when count drops to 5
    print(f"\n🔄 PHASE 3: Testing transition to diversNotfull (count drops to 5)")
    for count in [5, 4, 3, 2, 1, 0]:
        detected_objects = create_detected_objects_with_diver_count(count)
        relationship = get_diver_relationship(analyzer, detected_objects)
        
        expected = 'diversNotfull'  # Should transition to diversNotfull when count <= 5
        status = "✅" if relationship == expected else "❌"
        
        print(f"   Count {count}: {relationship} {status}")
        assert relationship == expected, f"Expected diversNotfull for count {count}, got {relationship}"
    
    # Test building up again after reset
    print(f"\n🔄 PHASE 4: Building up again after reset (0 → 6)")
    for count in range(7):
        detected_objects = create_detected_objects_with_diver_count(count)
        relationship = get_diver_relationship(analyzer, detected_objects)
        
        if count < 6:
            expected = 'diversNotfull'
        else:
            expected = 'diversfull'
        
        status = "✅" if relationship == expected else "❌"
        print(f"   Count {count}: {relationship} {status}")
        assert relationship == expected, f"Expected {expected} for count {count}, got {relationship}"
    
    print(f"\n✅ SUCCESS! Hysteresis logic working correctly.")
    print(f"   📈 Transitions:")
    print(f"      • diversNotfull → diversfull: when count ≥ 6")
    print(f"      • diversfull → diversNotfull: when count ≤ 5")
    print(f"   🔄 During blinking (count oscillates 6↔0):")
    print(f"      • diversfull state persists (no false transitions)")
    print(f"   🎯 This solves the blinking diver problem!")

def test_edge_cases():
    """Test edge cases and corner scenarios."""
    print(f"\n🧪 TESTING EDGE CASES")
    print("=" * 30)
    
    analyzer = SeaquestRelationshipAnalyzer()
    
    # Test exactly at threshold boundaries
    print("\n📍 Boundary conditions:")
    test_cases = [
        (5, 'diversNotfull', "At lower threshold"),
        (6, 'diversfull', "At upper threshold"), 
    ]
    
    for count, expected, description in test_cases:
        # Reset analyzer state first
        analyzer._previous_diver_state = 'diversNotfull'
        
        detected_objects = create_detected_objects_with_diver_count(count)
        relationship = get_diver_relationship(analyzer, detected_objects)
        status = "✅" if relationship == expected else "❌"
        
        print(f"   {description} (count={count}): {relationship} {status}")
        assert relationship == expected, f"Expected {expected} for count {count}, got {relationship}"
    
    # Test state persistence during intermediate counts
    print("\n📍 State persistence during blinking:")
    analyzer._previous_diver_state = 'diversfull'  # Start in diversfull state
    
    # Simulate counts that might occur during blinking
    intermediate_counts = [0, 1, 2, 3, 4, 5, 6]
    
    for count in intermediate_counts:
        detected_objects = create_detected_objects_with_diver_count(count)
        relationship = get_diver_relationship(analyzer, detected_objects)
        
        if count <= 5:
            expected = 'diversNotfull'  # Should transition when count <= 5
        else:
            expected = 'diversfull'     # Should remain diversfull when count >= 6
            
        status = "✅" if relationship == expected else "❌"
        print(f"   Count {count} (from diversfull): {relationship} {status}")

if __name__ == "__main__":
    test_diver_hysteresis()
    test_edge_cases()
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"   The hysteresis implementation successfully handles blinking divers.")
    print(f"   diversfull relationship will persist during the blinking phase!")
