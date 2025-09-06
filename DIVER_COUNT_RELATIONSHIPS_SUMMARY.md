# Diver Count Relationships Implementation

## Overview
Added `diversfull` and `diversNotfull` relationships to the Seaquest game that track the count of collected divers and determine the player's diver collection status.

## Relationships Added

### 1. diversfull
- **Condition**: When the count of collected divers is >= 6
- **Format**: `diversfull(player).`
- **DataFrame Format**: `diversfull(player)`

### 2. diversNotfull  
- **Condition**: When the count of collected divers is < 6
- **Format**: `diversNotfull(player).`
- **DataFrame Format**: `diversNotfull(player)`

## Implementation Details

### Files Modified

#### 1. `env/seaquest/relationship_analyzer.py`
- **Added Method**: `_analyze_diver_count_relationship()`
  - Counts objects of type `collected_diver`
  - Creates a virtual `diver_state` object to represent the diver count concept
  - Returns appropriate `SpatialRelationship` based on count

- **Updated Method**: `analyze_all_relationships()`
  - Added call to analyze diver count relationships
  - Relationships are added to the main relationship list

- **Updated Method**: `format_relationship_description()`
  - Added special formatting for `diver_state` object types
  - Returns clean format: `diversfull(player).` or `diversNotfull(player).`

- **Added Method**: `format_relationships_for_dataframe()`
  - Custom DataFrame formatting for Seaquest relationships
  - Handles special cases: water surface, diver state, facing side
  - Returns clean format without unnecessary object IDs

### Logic Flow

1. **Detection**: Count objects of type `'collected_diver'` in the detected objects dictionary
2. **Evaluation**: Compare count against threshold of 6
3. **Relationship Creation**: Create `SpatialRelationship` with virtual `diver_state` object
4. **Formatting**: Special formatting rules handle the virtual object appropriately

### Virtual Object Approach

Instead of creating a relationship without a second object, the implementation uses a virtual object:
```python
virtual_diver_state = GameObject('diver_state', (0, 0, 0, 0), object_id='diver_count_state')
```

This maintains consistency with the existing relationship system while representing conceptual states.

## Usage Examples

### Code Usage
```python
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer

analyzer = SeaquestRelationshipAnalyzer()

# Example with few divers
detected_objects = {
    'player': [GameObject('player', (100, 100, 20, 20), object_id='player_0')],
    'collected_diver': [
        GameObject('collected_diver', (50, 50, 10, 10), object_id='collected_diver_0'),
        GameObject('collected_diver', (60, 60, 10, 10), object_id='collected_diver_1'),
        # ... only 3 divers
    ]
}

relationships = analyzer.analyze_all_relationships(detected_objects)
# Output will include: SpatialRelationship(player_0, diver_count_state, 'diversNotfull')

# Get formatted description
descriptions = analyzer.get_relationship_descriptions(relationships)
# Output will include: "diversNotfull(player)."

# Get DataFrame format
df_format = analyzer.format_relationships_for_dataframe(relationships)
# Output will include: "diversNotfull(player)"
```

### Test Results
All test cases pass correctly:
- **0 divers** → `diversNotfull(player).`
- **3 divers** → `diversNotfull(player).`
- **6 divers** → `diversfull(player).`
- **8 divers** → `diversfull(player).`

## Integration

### With Existing System
- Works seamlessly with existing Seaquest relationship analysis
- Appears in complete relationship lists alongside spatial relationships
- Uses same formatting and description methods

### DataFrame Storage
- Clean format for pandas DataFrame storage
- No extraneous object IDs in the output
- Consistent with other Seaquest relationship formats

## Configuration

### Threshold Setting
The threshold of 6 divers is currently hardcoded in the `_analyze_diver_count_relationship()` method:
```python
if collected_diver_count >= 6:
    return SpatialRelationship(player, virtual_diver_state, 'diversfull')
```

This can be easily made configurable if needed by adding it to the Seaquest configuration.

### Extensibility
The virtual object approach allows for easy extension to other count-based or state-based relationships:
- Oxygen level relationships
- Score threshold relationships  
- Lives count relationships
- etc.

## Testing

Created comprehensive test suite in `test_diver_count_relationships.py` covering:
- Edge cases (0, 6, 8+ divers)
- Relationship detection accuracy
- Formatting correctness
- Integration with other relationships
- DataFrame formatting

All tests pass successfully, confirming correct implementation.
