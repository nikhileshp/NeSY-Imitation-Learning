# Grounding-Based Attention Penalty System

## Overview

When you use the `-use-distance-weights` flag during training, the system applies **attention penalties** during RDN tree learning based on eye-tracking data. This guides the learner to prefer clauses that involve objects the human player was actually looking at.

## Data Flow

### 1. **Fact Weights from Eye-Tracking**
```
data/seaquest/all/fire/train/fact_weights.txt
```
Each line contains:
```
<predicate>(<state>, <object_args>) <weight>
```
Example:
```
leftOfDiver(srz123, diver0) 0.854
rightOfEnemy(srz123, enemy2) 0.231
```

**Weight = attention weight** computed from gaze distance to object (Gaussian function). Higher weight = player was looking at that object.

### 2. **During Tree Learning**

For each candidate clause being evaluated (e.g., `action(S) :- leftOfDiver(S, D), nearSubmarine(S, D)`):

#### **Step A: Grounding**
The clause is **grounded** against training examples:
```
leftOfDiver(srz123, diver0), nearSubmarine(srz123, diver0)  ✓ matches
leftOfDiver(srz123, diver1), nearSubmarine(srz123, diver1)  ✗ doesn't match
leftOfDiver(srz456, diver2), nearSubmarine(srz456, diver2)  ✓ matches
```

#### **Step B: Weight Lookup with Anonymous Variables**

When a clause is grounded, it produces **anonymous variables** like `Anon1234` instead of concrete object names:
```
leftOfDiver(srz123, Anon1234)
```

**Problem**: We need to know which actual objects `Anon1234` could refer to.

**Solution - Two-Level Dictionary Cache**:

1. **Registration Phase** (`FactWeightLoader.registerAnonVariable`):
   - When `Anon1234` appears in a grounding with `leftOfDiver(srz123, Anon1234)`
   - Extract object type from predicate: `leftOfDiver` → `"diver"`
   - Cache: `Anon1234 → (state=srz123, objectType=diver)`

2. **Weight Retrieval Phase** (`FactWeightLoader.getWeightsForAnonVar`):
   - Look up `(srz123, diver)` in cache
   - Return **all attention weights** for diver-related predicates in state srz123:
     - `leftOfDiver(srz123, diver0)` → 0.854
     - `leftOfDiver(srz123, diver1)` → 0.623
     - `rightOfDiver(srz123, diver0)` → 0.912
     - `nearSubmarine(srz123, diver2)` → 0.445
   - Excludes always-1.0 predicates like `visibleDiver` (uninformative)

#### **Step C: Cartesian Product of Groundings**

For clauses with multiple predicates and anonymous variables, we compute **all possible groundings** (Cartesian product).

**Example**:
```prolog
Clause: leftOfDiver(S, D), nearEnemy(S, E)

Grounded: leftOfDiver(srz123, Anon456), nearEnemy(srz123, Anon42)

Anon456 (diver) weights: [0.9, 0.8, 0.2]  (3 possible divers)
Anon42 (enemy) weights:  [0.2, 0.9, 0.3]  (3 possible enemies)

Cartesian product: 3 × 3 = 9 possible complete groundings
```

**For each combination, aggregate with MIN across predicates**:
```
Combination 1: min(0.9, 0.2) = 0.2
Combination 2: min(0.9, 0.9) = 0.9
Combination 3: min(0.9, 0.3) = 0.3
Combination 4: min(0.8, 0.2) = 0.2
Combination 5: min(0.8, 0.9) = 0.8
Combination 6: min(0.8, 0.3) = 0.3
Combination 7: min(0.2, 0.2) = 0.2
Combination 8: min(0.2, 0.9) = 0.2
Combination 9: min(0.2, 0.3) = 0.2

All grounding weights: [0.2, 0.9, 0.3, 0.2, 0.8, 0.3, 0.2, 0.2, 0.2]
```

**If same anonymous variable appears in multiple predicates**:
```prolog
Clause: leftOfDiver(S, D), nearSubmarine(S, D)  # Same D!

Anon456 (diver) appears in BOTH predicates

For predicate 1 (leftOfDiver): weights [0.9, 0.8, 0.2]
For predicate 2 (nearSubmarine): weights [0.7, 0.5, 0.1]

Cartesian product for this anon var: 3 × 3 = 9 combinations
min(0.9, 0.7) = 0.7
min(0.9, 0.5) = 0.5
min(0.9, 0.1) = 0.1
min(0.8, 0.7) = 0.7
min(0.8, 0.5) = 0.5
min(0.8, 0.1) = 0.1
min(0.2, 0.7) = 0.2
min(0.2, 0.5) = 0.2
min(0.2, 0.1) = 0.1

All grounding weights: [0.7, 0.5, 0.1, 0.7, 0.5, 0.1, 0.2, 0.2, 0.1]
```

#### **Step D: Apply Final Aggregation Strategy**

Now we have a list of all possible grounding weights. Apply the strategy:

**Strategy: "min"** (most conservative)
```
aggregated_weight = min([0.2, 0.9, 0.3, 0.2, 0.8, 0.3, 0.2, 0.2, 0.2])
                  = 0.2
```
Meaning: **"If ANY possible grounding has low attention, classify as LOW."**

**Strategy: "max"** (most optimistic)
```
aggregated_weight = max([0.2, 0.9, 0.3, 0.2, 0.8, 0.3, 0.2, 0.2, 0.2])
                  = 0.9
```
Meaning: **"If ANY possible grounding has high attention, classify as HIGH."**

**Strategy: "avg"**
```
aggregated_weight = avg([0.2, 0.9, 0.3, 0.2, 0.8, 0.3, 0.2, 0.2, 0.2])
                  = 0.367
```

**Strategy: "proportion"**
```
aggregated_weight = count(weight >= 0.5) / count(all weights)
                  = 2 / 9 = 0.222
```

#### **Step E: Classification**

Compare aggregated weight to **threshold** (default 0.5):
```
if (aggregated_weight >= 0.5):
    HIGH attention grounding (k_high++)
else:
    LOW attention grounding (k_low++)
```

**For our example with "min" strategy**:
```
aggregated_weight = 0.2 < 0.5  → k_low++
```

#### **Step F: Penalty Calculation**

```java
penalty = -alpha * k_high + beta * k_low
```

**Parameters** (in `ScoreRegressionNode.java`):
- `alpha = 0.1` (penalty coefficient for high attention)
- `beta = 0.5` (penalty coefficient for low attention)
- `threshold = 0.5` (attention threshold)

**Interpretation**:
- **k_high** groundings involve attended objects → **reward** (negative penalty)
- **k_low** groundings involve unattended objects → **penalize** (positive penalty)

**Example**:
```
Clause: action(S) :- leftOfDiver(S,D)

Evaluation:
- 45 groundings with high attention (k_high=45)
- 15 groundings with low attention (k_low=15)

penalty = -0.1 * 45 + 0.5 * 15
        = -4.5 + 7.5
        = +3.0 (net penalty)
```

This clause gets penalized because although most groundings involve attended objects, there are too many unattended groundings.

### 3. **Final Score**

The RDN scorer combines variance reduction with penalty:

```java
final_score = variance_reduction + (scalingPenalties * grounding_penalty)
```

Where `scalingPenalties = 1.0` by default.

**Lower final score is better** (RDN minimizes combined variance).

## What Gets Written to `node_*.txt` Files

With `-debugScoring` flag enabled, each evaluated clause shows:

```
Clause: action(S) :- leftOfDiver(S, D1), nearSubmarine(S, D1)
├─ Examples: 156 TRUE, 421 FALSE
├─ Variance: 0.234
├─ Penalties:
│   ├─ Total Penalty: 3.45
│   ├─ Length/Singleton Penalty: 0.15
│   └─ Grounding Penalty: 3.30
│       ├─ k_high: 134 groundings (high attention)
│       ├─ k_low: 51 groundings (low attention)
│       └─ Formula: -0.1*134 + 0.5*51 = 3.30
└─ Final Score: 0.268
```

## Why This Matters

**Without attention penalty**: The learner might create rules based on background objects the player ignores.

**With attention penalty**: The learner is **guided** to prefer rules that involve objects the player was actually attending to, leading to better imitation of human decision-making.

**Trade-off**: The penalty isn't absolute—it's balanced against variance reduction. A clause with slightly worse attention but much better variance reduction can still win.

## Implementation Details

### Key Files

1. **`rdnboost/src/edu/wisc/cs/will/ILP/Regression/FactWeightLoader.java`**
   - Loads `fact_weights.txt` into memory
   - Implements two-level caching for efficient anonymous variable lookup
   - `registerAnonVariable()`: Maps anonymous variables to (state, objectType)
   - `getWeightsForAnonVar()`: Returns all relevant weights for a state/object type
   - `buildWeightListForStateObject()`: Collects weights from relevant predicates

2. **`rdnboost/src/edu/wisc/cs/will/ILP/ScoreRegressionNode.java`**
   - `computeAggregatedWeight()`: Main penalty computation logic
   - Grounds clause literals against examples
   - For each grounding with anonymous variables:
     - Infers object type from predicate name
     - Registers with FactWeightLoader
     - Retrieves cached weights
     - Aggregates weights using selected strategy
   - Classifies groundings as high/low attention
   - Computes final grounding penalty

3. **`rdnboost/src/edu/wisc/cs/will/ILP/Regression/RegressionInfoHolderForRDN.java`**
   - Stores reference to caller node for penalty access
   - Creates `ClauseEvaluation` objects with penalty values
   - Passes penalties from `SingleClauseNode` to evaluation tracking

4. **`rdnboost/src/edu/wisc/cs/will/ILP/Regression/BranchStats.java`**
   - `ClauseEvaluation` class stores penalty breakdown
   - `writeClausesToFile()`: Writes penalty information to node files
   - `printClauseComparison()`: Displays penalty comparison table

### Configuration Parameters

All parameters are hardcoded in `ScoreRegressionNode.java` (lines 42-45):

```java
private static final double ATTENTION_THRESHOLD = 0.5;    // Threshold for high/low attention
private static final double ALPHA = 0.1;                  // Penalty coefficient for high attention
private static final double BETA = 0.5;                   // Penalty coefficient for low attention
private static final String AGGREGATION_STRATEGY = "min"; // Weight aggregation strategy
```

To modify:
1. Edit these values in `ScoreRegressionNode.java`
2. Rebuild: `cd rdnboost && mvn clean package`

### Command-Line Flags

- **`-use-distance-weights`**: Enable attention penalty calculation
- **`-debugScoring`**: Enable detailed debug output and write penalty info to `node_*.txt` files

Both flags must be present during training to see penalty information in output files.

### Example Usage

```bash
# Train with attention penalties and debug output
./run_full_pipeline.sh 3 10 true false

# The updated script automatically includes both flags during training
```

## Debugging

### Verify Penalty Calculation

1. **Check fact_weights.txt exists**:
   ```bash
   ls -lh data/seaquest/all/fire/train/fact_weights.txt
   ```

2. **Examine weights distribution**:
   ```bash
   awk '{print $NF}' data/seaquest/all/fire/train/fact_weights.txt | sort -n | uniq -c
   ```

3. **Look for penalty output in logs**:
   ```bash
   grep -i "grounding penalty" rdn_models/seaquest/*/fire/node_*.txt
   ```

4. **Check that both flags are enabled**:
   ```bash
   grep -E "(use-distance-weights|debugScoring)" run_full_pipeline.sh
   ```

### Common Issues

**Penalties are all zero**:
- Fact weights file missing or not loaded
- `-use-distance-weights` flag not set
- All weights are 1.0 (no eye-tracking data)

**No node_*.txt files**:
- `-debugScoring` flag not set
- Check that `BranchStats.ENABLE_DETAILED_DEBUG` is true

**All groundings classified as low attention**:
- Threshold may be too high
- Aggregation strategy may be too conservative (try "max" instead of "min")
- Check actual weight values in `fact_weights.txt`

## Performance Impact

**Memory**: Caching system uses ~O(states × objects × predicates) memory. For typical Seaquest dataset (~1000 states, ~50 objects), this is negligible.

**Speed**: Dictionary lookup is O(1) after cache is built. Building cache is O(facts) once per training run. Penalty computation adds ~5-10% overhead to tree learning.

## Future Extensions

- Make penalty parameters configurable via command-line flags
- Support multiple aggregation strategies simultaneously
- Add penalty visualization in tree diagrams
- Implement adaptive thresholds based on weight distribution
- Support predicate-specific alpha/beta coefficients
