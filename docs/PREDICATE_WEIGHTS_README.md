# Per-Relationship Predicate Weights

## Overview

The `--euclidean-distance-weights` flag now computes **per-relationship predicate weights** based on eye-tracking data. Each relationship in the `relationships` column gets a corresponding weight in the `predicate_weights` column.

## How It Works

### 1. Object-Level Attention Weights

First, attention weights are computed for each detected object based on the Euclidean distance from the gaze point to the object's centroid:

```python
weight = s / (distance + s)
where s = 0.75 * min(frame_width, frame_height)
```

Objects closer to the gaze point get higher weights (approaching 1.0), while distant objects get lower weights (approaching 0.0).

### 2. Relationship-Level Predicate Weights

Each relationship is then assigned a weight based on its type:

#### Non-Grounded Relationships (Weight = 1.0)
These relationships don't involve specific game objects:
- **Facing relationships**: `facingRight()`, `facingLeft()`, `enemyFacingRight(enemy1)`
- **Water surface**: `aboveWater()`, `belowWater()`
- **Diver state**: `diversfull()`, `diversNotfull()`, `diversEmpty()`
- **Oxygen state**: `oxygenLow()`, `oxygenOk()`
- **Visibility**: `visibleEnemy(enemy1)`, `visibleDiver(diver1)`

#### Grounded Relationships (Weight = Object Attention Weight)
These relationships involve specific objects and use the attention weight of the grounded object:
- **Spatial relationships**: `nearbyDiver(diver1)`, `leftOfEnemy(enemy2)`, `aboveEnemy(enemy3)`
- The weight comes from the attention weight of the grounded object (e.g., `diver1`, `enemy2`)

### 3. Example Weight

The example weight is calculated as:
```python
example_weight = 1.0 + 5.0 * max(all_relationship_weights)
```

This gives higher weights to examples where the player is looking at relevant objects.

## CSV Format

### Relationships Column
Space-separated relationships with commas and trailing comma:
```
facingRight , nearbyDiver(diver1) , leftOfEnemy(enemy1) , aboveWater , 
```

### Predicate Weights Column
Space-separated weights in the **same order** as relationships:
```
1.0000 0.8523 0.3421 1.0000
```

### Alignment Example
```
Relationships:     facingRight , nearbyDiver(diver1) , leftOfEnemy(enemy1) , aboveWater , 
Predicate Weights: 1.0000       0.8523                0.3421                 1.0000
```

## Usage

```bash
# Process with per-relationship predicate weights
python main.py --data path/to/trajectory \
  --euclidean-distance-weights \
  --no-visual --process-all

# Process all trajectories
python main.py --data path/to/parent_folder \
  --all-trajectories \
  --euclidean-distance-weights \
  --no-visual --process-all
```

## Output Files

### Individual Trajectory
Each trajectory folder gets updated with:
- `<trajectory_name>.txt` → `<trajectory_name>_with_relationships_and_goals.txt`

### All Trajectories
When using `--all-trajectories`, a consolidated file is created:
- `relationships.txt` in the parent folder

Both files contain columns:
- `frameid`: Frame identifier
- `episode_id`: Episode ID
- `score`: Game score
- `duration`: Frame duration
- `unclipped_reward`: Reward value
- `action`: Action taken (0-17 for Seaquest)
- `objects`: Detected objects
- `relationships`: Space-separated relationships
- `goal`: Detected goal
- `distance_weights`: (empty when using euclidean weights)
- `predicate_weights`: Space-separated weights (one per relationship)
- `example_weight`: Overall example weight
- `trajectory`: Trajectory name (only in consolidated file)

## Implementation Details

### Key Functions

#### `main.py`
- `_compute_relationship_predicate_weights(relationships, object_weight_map)`: Assigns weights to each relationship
- Returns list of weights in same order as relationships list

#### `attention_weights.py`
- `calculate_predicate_weights(eye_pos, centroids, width, height, k)`: Computes object-level attention weights
- `create_object_weight_mapping(detected_objects, object_types, predicate_weights)`: Maps object IDs to weights
- `calculate_example_weight(predicate_weights)`: Computes overall example weight

### Order Preservation

**Critical**: The weights list maintains the same order as the relationships list:
1. Relationships are analyzed in order by `analyze_all_relationships()`
2. Weights are computed in the same loop iteration order
3. Both are formatted to strings in the same order
4. The CSV columns align perfectly

## For RDN Training

When generating `fact_weights.txt` for use with the Java RDN implementation:

### Step 1: Parse the CSV
```python
import pandas as pd

df = pd.read_csv('relationships.txt')

for _, row in df.iterrows():
    frame_id = row['frameid']
    relationships = row['relationships'].strip(' ,').split(' , ')
    weights = row['predicate_weights'].split()
    
    # Create fact_weights.txt entries
    for rel, weight in zip(relationships, weights):
        # rel is like "nearbyDiver(diver1)" or "facingRight"
        # weight is like "0.8523"
        print(f"{rel}. {weight}")
```

### Step 2: Generate fact_weights.txt
```
nearbyDiver(diver1). 0.8523
leftOfEnemy(enemy1). 0.3421
facingRight. 1.0000
aboveWater. 1.0000
```

This file can then be loaded by the Java RDN implementation using the `--use-distance-weights` flag.

## Benefits

1. **Per-relationship precision**: Each relationship gets its own attention weight
2. **Context-aware**: Relationships involving attended objects are weighted higher
3. **Flexible**: Non-grounded relationships (like facing direction) are preserved with neutral weight
4. **RDN-compatible**: Format aligns with the Java implementation's expected input

## Troubleshooting

### Weights Don't Match Relationships
- **Symptom**: Number of weights ≠ number of relationships
- **Cause**: Logic error in weight computation
- **Fix**: Ensure `_compute_relationship_predicate_weights` returns exactly one weight per relationship

### All Weights Are 1.0
- **Symptom**: No variation in weights
- **Cause**: Object weight map is empty or gaze positions not available
- **Fix**: Check that gaze data is loaded and objects are detected

### Weights Out of Order
- **Symptom**: High weight assigned to non-attended objects
- **Cause**: Order mismatch between relationships and weights
- **Fix**: Verify that both lists are built in the same iteration order (already fixed in current implementation)
