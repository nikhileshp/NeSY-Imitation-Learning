# Distance Weight Calculation

This document explains the distance weight calculation feature that computes how strongly gaze positions relate to spatial objects in relationships.

## Overview

The distance weight feature calculates weights for relationships involving spatial objects (divers, enemies, enemy submarines, enemy missiles) based on the proximity of gaze coordinates to object centers. The weight formula is:

```
distance_weight = max_possible_distance / actual_distance
```

Where:
- `max_possible_distance` = √(screen_width² + screen_height²)
- `actual_distance` = Euclidean distance between gaze position and object center

## Key Features

- **Automatic calculation**: Distance weights are automatically calculated during main processing
- **Individual relationships**: Each relationship instance gets its own weight based on the displayed gaze position
- **Targeted objects**: Only calculates weights for relationships involving divers, enemies, enemy submarines, and enemy missiles
- **Enhanced visualization**: Gaze positions are displayed prominently with distance weight information
- **DataFrame storage**: Formatted weights are stored in the gaze DataFrame for analysis

## Usage

### Command Line Processing

When running the main analysis pipeline, distance weights are automatically calculated if gaze data is available:

```bash
python main.py --data /path/to/trajectory --verbose 2 --save-rel
```

The `--verbose 2` flag will show distance weight calculations in the output.

### Accessing Distance Weights

Distance weights are stored in a single column in the saved gaze DataFrame:

- `distance_weights`: Individual relationship weights calculated from the displayed gaze position (last gaze position in frame)

Format: `relationshipIdentifier(objectId):weight ; relationshipIdentifier(objectId):weight ; ...`

Example:
```
nearbyDiver(diver_1):528.02 ; leftOfEnemy(enemy_1):7.80 ; visibleEnemySubmarine(sub_1):15.00
```

### Programmatic Usage

```python
from core.distance_weight_calculator import DistanceWeightCalculator

# Initialize with screen dimensions
calculator = DistanceWeightCalculator(width=640, height=480)

# Calculate individual relationship weights (uses last gaze position)
distance_weights = calculator.calculate_relationship_distance_weights(
    relationships, gaze_positions
)

# Format for DataFrame storage
formatted = calculator.format_distance_weights_for_dataframe(distance_weights)

# distance_weights is now a dict like:
# {"nearbyDiver(diver_1)": 10.5, "leftOfEnemy(enemy_1)": 7.8, ...}
```

## Target Object Types

Distance weights are calculated for relationships involving these object types:

- `diver`: Divers to be collected
- `enemy`: Regular enemies
- `enemy_submarine`: Enemy submarines
- `enemy_missile`: Enemy missiles

## Relationship Types

Common relationship types that receive distance weights:

- `nearbyDiver`, `leftOfDiver`, `rightOfDiver`, etc.
- `nearbyEnemy`, `leftOfEnemy`, `belowOfEnemy`, etc.
- `visibleDiver`, `visibleEnemy`, `visibleEnemySubmarine`
- `enemyFacingLeft`, `enemyFacingRight`

## Weight Interpretation

- **Higher weights**: Indicate gaze positions closer to objects (more attention)
- **Lower weights**: Indicate gaze positions farther from objects (less attention)
- **Maximum weight**: Equals max_possible_distance when gaze is exactly on object center
- **Minimum weight approaches 0**: As gaze moves to opposite corners of screen

## Examples

See `test_distance_weights.py` and `example_distance_weights.py` for comprehensive examples of the distance weight calculation system in action.

## File Locations

- **Core implementation**: `core/distance_weight_calculator.py`
- **Integration**: `main.py` (processing pipeline)
- **Data storage**: `core/gaze_data_processor.py` (DataFrame columns)
- **Tests**: `test_distance_weights.py`
- **Examples**: `example_distance_weights.py`