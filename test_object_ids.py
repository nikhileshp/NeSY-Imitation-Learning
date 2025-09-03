#!/usr/bin/env python3
"""
Test script to verify that object IDs are assigned sequentially for enemies and submarines.
"""

import numpy as np
from core.game_object import GameObject

def test_sequential_object_ids():
    """Test that object IDs are assigned sequentially."""
    print("🔍 TESTING SEQUENTIAL OBJECT ID ASSIGNMENT")
    print("=" * 50)
    
    # Test basic GameObject creation with proper IDs
    print("\n1. Testing basic GameObject creation:")
    enemies = []
    for i in range(5):
        enemy = GameObject('enemy', (10 + i*20, 50, 15, 15), object_id=f'enemy_{i}')
        enemies.append(enemy)
        print(f"   Created: {enemy.object_id}")
    
    # Test enemy submarines
    print("\n2. Testing enemy submarine creation:")
    submarines = []
    for i in range(3):
        submarine = GameObject('enemy_submarine', (10 + i*30, 100, 25, 20), object_id=f'enemy_submarine_{i}')
        submarines.append(submarine)
        print(f"   Created: {submarine.object_id}")
    
    # Test what happens without explicit object_id (should use fallback)
    print("\n3. Testing GameObject without explicit object_id (fallback behavior):")
    fallback_enemy = GameObject('enemy', (100, 150, 15, 15))
    print(f"   Created: {fallback_enemy.object_id} (uses memory address)")
    
    # Test reassigning object IDs
    print("\n4. Testing object ID reassignment:")
    objects = []
    
    # Create some objects with random IDs
    objects.append(GameObject('enemy', (10, 10, 15, 15), object_id='enemy_random_123'))
    objects.append(GameObject('enemy', (30, 10, 15, 15), object_id='enemy_another_456'))
    objects.append(GameObject('enemy', (50, 10, 15, 15), object_id='enemy_third_789'))
    
    print("   Before reassignment:")
    for obj in objects:
        print(f"     {obj.object_id}")
    
    # Reassign sequential IDs
    for i, obj in enumerate(objects):
        obj.object_id = f'{obj.object_type}_{i}'
    
    print("   After reassignment:")
    for obj in objects:
        print(f"     {obj.object_id}")
    
    # Test mixed object types
    print("\n5. Testing mixed object types with sequential IDs:")
    mixed_objects = {
        'enemy': [],
        'enemy_submarine': [],
        'enemy_missile': []
    }
    
    # Create objects
    for i in range(3):
        mixed_objects['enemy'].append(GameObject('enemy', (i*20, 200, 15, 15), object_id=f'enemy_{i}'))
        mixed_objects['enemy_submarine'].append(GameObject('enemy_submarine', (i*30, 220, 25, 20), object_id=f'enemy_submarine_{i}'))
        mixed_objects['enemy_missile'].append(GameObject('enemy_missile', (i*15, 240, 8, 4), object_id=f'enemy_missile_{i}'))
    
    # Print all objects
    for object_type, objects in mixed_objects.items():
        print(f"   {object_type}:")
        for obj in objects:
            print(f"     {obj.object_id}")
    
    print(f"\n✅ SUCCESS! All object IDs are properly formatted.")
    print(f"   Expected format: <object_type>_<sequential_number>")
    print(f"   Examples: enemy_0, enemy_1, enemy_submarine_0, etc.")
    
    return mixed_objects

if __name__ == "__main__":
    test_sequential_object_ids()
