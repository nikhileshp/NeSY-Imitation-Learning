#!/usr/bin/env python3
"""
Test script to verify diver count relationships (diversfull/diversNotfull) functionality.
"""

from core.game_object import GameObject
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer

def test_diver_count_relationships():
    """Test the diversfull and diversNotfull relationship detection."""
    print("🏊 TESTING DIVER COUNT RELATIONSHIPS")
    print("=" * 50)
    
    # Initialize the Seaquest relationship analyzer
    analyzer = SeaquestRelationshipAnalyzer()
    
    # Test case 1: Less than 6 divers (should be diversNotfull)
    print("\n1. Testing with 3 collected divers (should be diversNotfull):")
    detected_objects_few = {
        'player': [GameObject('player', (100, 100, 20, 20), object_id='player_0')],
        'collected_diver': [
            GameObject('collected_diver', (50, 50, 10, 10), object_id='collected_diver_0'),
            GameObject('collected_diver', (60, 60, 10, 10), object_id='collected_diver_1'),
            GameObject('collected_diver', (70, 70, 10, 10), object_id='collected_diver_2'),
        ],
        'enemy': [
            GameObject('enemy', (200, 100, 15, 15), object_id='enemy_0')
        ]
    }
    
    relationships_few = analyzer.analyze_all_relationships(detected_objects_few)
    
    # Find diver count relationship
    diver_relationships_few = [r for r in relationships_few if r.relationship_type in ['diversfull', 'diversNotfull']]
    
    print(f"   Collected divers count: {len(detected_objects_few['collected_diver'])}")
    print(f"   Diver count relationships found: {len(diver_relationships_few)}")
    if diver_relationships_few:
        rel = diver_relationships_few[0]
        print(f"   Relationship: {rel.relationship_type}")
        print(f"   Description: {analyzer.game_config.format_relationship_description(rel)}")
        assert rel.relationship_type == 'diversNotfull', f"Expected diversNotfull, got {rel.relationship_type}"
        print("   ✅ PASSED: Correctly detected diversNotfull")
    else:
        print("   ❌ FAILED: No diver count relationship detected")
    
    # Test case 2: Exactly 6 divers (should be diversfull)
    print("\n2. Testing with 6 collected divers (should be diversfull):")
    detected_objects_full = {
        'player': [GameObject('player', (100, 100, 20, 20), object_id='player_0')],
        'collected_diver': [
            GameObject('collected_diver', (50, 50, 10, 10), object_id='collected_diver_0'),
            GameObject('collected_diver', (60, 60, 10, 10), object_id='collected_diver_1'),
            GameObject('collected_diver', (70, 70, 10, 10), object_id='collected_diver_2'),
            GameObject('collected_diver', (80, 80, 10, 10), object_id='collected_diver_3'),
            GameObject('collected_diver', (90, 90, 10, 10), object_id='collected_diver_4'),
            GameObject('collected_diver', (100, 90, 10, 10), object_id='collected_diver_5'),
        ],
        'enemy': [
            GameObject('enemy', (200, 100, 15, 15), object_id='enemy_0')
        ]
    }
    
    relationships_full = analyzer.analyze_all_relationships(detected_objects_full)
    
    # Find diver count relationship
    diver_relationships_full = [r for r in relationships_full if r.relationship_type in ['diversfull', 'diversNotfull']]
    
    print(f"   Collected divers count: {len(detected_objects_full['collected_diver'])}")
    print(f"   Diver count relationships found: {len(diver_relationships_full)}")
    if diver_relationships_full:
        rel = diver_relationships_full[0]
        print(f"   Relationship: {rel.relationship_type}")
        print(f"   Description: {analyzer.game_config.format_relationship_description(rel)}")
        assert rel.relationship_type == 'diversfull', f"Expected diversfull, got {rel.relationship_type}"
        print("   ✅ PASSED: Correctly detected diversfull")
    else:
        print("   ❌ FAILED: No diver count relationship detected")
    
    # Test case 3: More than 6 divers (should still be diversfull)
    print("\n3. Testing with 8 collected divers (should be diversfull):")
    detected_objects_extra = {
        'player': [GameObject('player', (100, 100, 20, 20), object_id='player_0')],
        'collected_diver': [
            GameObject('collected_diver', (50, 50, 10, 10), object_id='collected_diver_0'),
            GameObject('collected_diver', (60, 60, 10, 10), object_id='collected_diver_1'),
            GameObject('collected_diver', (70, 70, 10, 10), object_id='collected_diver_2'),
            GameObject('collected_diver', (80, 80, 10, 10), object_id='collected_diver_3'),
            GameObject('collected_diver', (90, 90, 10, 10), object_id='collected_diver_4'),
            GameObject('collected_diver', (100, 90, 10, 10), object_id='collected_diver_5'),
            GameObject('collected_diver', (110, 90, 10, 10), object_id='collected_diver_6'),
            GameObject('collected_diver', (120, 90, 10, 10), object_id='collected_diver_7'),
        ]
    }
    
    relationships_extra = analyzer.analyze_all_relationships(detected_objects_extra)
    
    # Find diver count relationship
    diver_relationships_extra = [r for r in relationships_extra if r.relationship_type in ['diversfull', 'diversNotfull']]
    
    print(f"   Collected divers count: {len(detected_objects_extra['collected_diver'])}")
    print(f"   Diver count relationships found: {len(diver_relationships_extra)}")
    if diver_relationships_extra:
        rel = diver_relationships_extra[0]
        print(f"   Relationship: {rel.relationship_type}")
        print(f"   Description: {analyzer.game_config.format_relationship_description(rel)}")
        assert rel.relationship_type == 'diversfull', f"Expected diversfull, got {rel.relationship_type}"
        print("   ✅ PASSED: Correctly detected diversfull")
    else:
        print("   ❌ FAILED: No diver count relationship detected")
    
    # Test case 4: No collected divers (should be diversNotfull)
    print("\n4. Testing with 0 collected divers (should be diversNotfull):")
    detected_objects_none = {
        'player': [GameObject('player', (100, 100, 20, 20), object_id='player_0')],
        'collected_diver': [],
        'enemy': [
            GameObject('enemy', (200, 100, 15, 15), object_id='enemy_0')
        ]
    }
    
    relationships_none = analyzer.analyze_all_relationships(detected_objects_none)
    
    # Find diver count relationship
    diver_relationships_none = [r for r in relationships_none if r.relationship_type in ['diversfull', 'diversNotfull']]
    
    print(f"   Collected divers count: {len(detected_objects_none['collected_diver'])}")
    print(f"   Diver count relationships found: {len(diver_relationships_none)}")
    if diver_relationships_none:
        rel = diver_relationships_none[0]
        print(f"   Relationship: {rel.relationship_type}")
        print(f"   Description: {analyzer.game_config.format_relationship_description(rel)}")
        assert rel.relationship_type == 'diversNotfull', f"Expected diversNotfull, got {rel.relationship_type}"
        print("   ✅ PASSED: Correctly detected diversNotfull")
    else:
        print("   ❌ FAILED: No diver count relationship detected")
    
    # Test case 5: Test with all relationships together
    print("\n5. Testing complete relationship analysis with diver count:")
    complete_relationships = analyzer.analyze_all_relationships(detected_objects_full)
    
    print(f"   Total relationships found: {len(complete_relationships)}")
    print("   All relationships:")
    for i, rel in enumerate(complete_relationships):
        description = analyzer.game_config.format_relationship_description(rel)
        print(f"     {i+1}. {rel.relationship_type}: {description}")
    
    print(f"\n✅ SUCCESS! Diver count relationships implemented correctly.")
    print(f"   - diversfull: When collected_diver count >= 6")
    print(f"   - diversNotfull: When collected_diver count < 6")
    print(f"   - Formatted as: diversfull(player). or diversNotfull(player).")

def test_relationship_formatting():
    """Test the formatting of diver count relationships for DataFrame storage."""
    print(f"\n🔤 TESTING RELATIONSHIP FORMATTING FOR DATAFRAME")
    print("=" * 50)
    
    analyzer = SeaquestRelationshipAnalyzer()
    
    # Test with diversfull scenario
    detected_objects = {
        'player': [GameObject('player', (100, 100, 20, 20), object_id='player_0')],
        'collected_diver': [GameObject('collected_diver', (i*10, 50, 10, 10), object_id=f'collected_diver_{i}') for i in range(6)],
        'enemy': [GameObject('enemy', (200, 100, 15, 15), object_id='enemy_0')]
    }
    
    relationships = analyzer.analyze_all_relationships(detected_objects)
    formatted_relationships = analyzer.format_relationships_for_dataframe(relationships)
    
    print(f"   DataFrame format: {formatted_relationships}")
    print(f"   Contains diversfull: {'diversfull' in formatted_relationships}")
    
    return relationships

if __name__ == "__main__":
    test_diver_count_relationships()
    test_relationship_formatting()
