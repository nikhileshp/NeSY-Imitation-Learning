# Grounding-Based Attention Penalty for RDN Clause Scoring

## Overview

This feature adds a penalty term to RDN clause scoring that considers the attention weights (from eye-tracking data) of **all possible groundings** of a clause, not just the weighted average. This allows the system to prefer clauses where:
- Most or all groundings involve attended objects (high attention weights)
- Groundings that involve unattended objects are penalized

## How It Works

### Scoring Formula

During clause construction, for each candidate clause:

1. **Ground the clause** for each training example (using binding lists)
2. **For each grounding** (set of grounded predicates):
   - Aggregate attention weights of all predicates in the grounding
   - Compare aggregated weight to threshold
3. **Count**:
   - `k_high` = number of groundings with weight ≥ threshold
   - `k_low` = number of groundings with weight < threshold
4. **Compute penalty**: 
   ```
   grounding_penalty = -alpha * k_high + beta * k_low
   ```
   - Negative contribution (reward) for attended groundings
   - Positive contribution (penalty) for unattended groundings

### Aggregation Strategies

When a grounding involves multiple predicates, their weights are aggregated using one of these strategies:

#### 1. **`min` (Conservative)** ⭐ Recommended
```java
aggregated_weight = min(weight_1, weight_2, ..., weight_n)
```
- A grounding is only "attended" if **ALL** its predicates have high weights
- Use when: You want clauses where the entire conjunction was attended to
- Example: `nearby(submarine, diver1) AND oxygen_low(state)` 
  - Only counts as "attended" if both predicates have high weights

#### 2. **`max` (Optimistic)**
```java
aggregated_weight = max(weight_1, weight_2, ..., weight_n)
```
- A grounding is "attended" if **ANY** predicate has high weight
- Use when: You want to reward clauses that involve any attended objects
- Risk: May select clauses with only partially attended groundings

#### 3. **`avg` (Balanced)**
```java
aggregated_weight = (weight_1 + weight_2 + ... + weight_n) / n
```
- Average attention across all predicates
- Use when: You want a moderate approach
- Allows some unattended predicates if others are highly attended

#### 4. **`proportion` (Fine-grained)**
```java
aggregated_weight = count(weights >= threshold) / total_predicates
```
- Returns the proportion of predicates above threshold
- Use when: You want continuous scoring based on how many predicates are attended
- Example: If 2 out of 3 predicates are attended, returns 0.67

## Configuration

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Attention weight threshold for considering a grounding "attended" |
| `alpha` | 0.1 | Reward coefficient per high-attention grounding (reduces penalty) |
| `beta` | 0.5 | Penalty coefficient per low-attention grounding (increases penalty) |
| `strategy` | `"min"` | Aggregation strategy: `"min"`, `"max"`, `"avg"`, or `"proportion"` |

### Setting Parameters

#### Option 1: Java System Properties (Recommended)
```bash
java -Dgrounding.penalty.threshold=0.6 \
     -Dgrounding.penalty.alpha=0.2 \
     -Dgrounding.penalty.beta=0.3 \
     -Dgrounding.penalty.strategy=min \
     -jar rdnboost/target/boostsrl-weights-2.0.0.jar [args...]
```

#### Option 2: Modify Pipeline Script
In `run_full_pipeline.sh`, add system properties to the Java command:
```bash
JAVA_OPTS="-Dgrounding.penalty.threshold=0.6"
JAVA_OPTS="$JAVA_OPTS -Dgrounding.penalty.alpha=0.2"
JAVA_OPTS="$JAVA_OPTS -Dgrounding.penalty.beta=0.3"
JAVA_OPTS="$JAVA_OPTS -Dgrounding.penalty.strategy=avg"

java $JAVA_OPTS -jar "$JAR" -l -train "$DATA_DIR/$action/train/" -target "$action" ...
```

#### Option 3: Environment Variables (requires shell wrapper)
```bash
export GROUNDING_PENALTY_THRESHOLD=0.6
export GROUNDING_PENALTY_ALPHA=0.2
export GROUNDING_PENALTY_BETA=0.3
export GROUNDING_PENALTY_STRATEGY=min
```

## Usage

The grounding penalty is **automatically enabled** when:
1. `fact_weights.txt` is present in the training directory
2. The `-use-distance-weights` flag is set (or equivalent)

No additional flags are needed—the presence of `fact_weights.txt` triggers the penalty computation.

### Example Run

```bash
# Train with grounding penalty using default parameters
java -Dgrounding.penalty.threshold=0.5 \
     -jar rdnboost/target/boostsrl-weights-2.0.0.jar \
     -l -train data/seaquest/all/fire/train/ \
     -target fire -trees 10 -depth 3

# Train with custom parameters and "avg" aggregation
java -Dgrounding.penalty.threshold=0.7 \
     -Dgrounding.penalty.alpha=0.15 \
     -Dgrounding.penalty.beta=0.4 \
     -Dgrounding.penalty.strategy=avg \
     -jar rdnboost/target/boostsrl-weights-2.0.0.jar \
     -l -train data/seaquest/all/fire/train/ \
     -target fire -trees 10 -depth 3
```

## Tuning Guidelines

### Threshold (`threshold`)
- **Lower (0.3-0.5)**: More lenient, allows moderate attention
- **Medium (0.5-0.7)**: Balanced, requires clear attention signal
- **Higher (0.7-0.9)**: Strict, only very high attention counts as "attended"

### Alpha Reward (`alpha`)
- **Low (0.05-0.15)**: Small reward for attended groundings
- **Medium (0.15-0.3)**: Moderate preference for attended groundings
- **High (0.3-0.5)**: Strong preference for attended groundings

### Beta Penalty (`beta`)
- **Low (0.1-0.3)**: Tolerates some unattended groundings
- **Medium (0.3-0.6)**: Balanced penalty for unattended groundings
- **High (0.6-1.0)**: Strongly penalizes unattended groundings

### Recommended Starting Points

#### Conservative (prefer clauses where all groundings are attended)
```bash
-Dgrounding.penalty.threshold=0.6
-Dgrounding.penalty.alpha=0.2
-Dgrounding.penalty.beta=0.5
-Dgrounding.penalty.strategy=min
```

#### Balanced (reward attended, penalize unattended proportionally)
```bash
-Dgrounding.penalty.threshold=0.5
-Dgrounding.penalty.alpha=0.15
-Dgrounding.penalty.beta=0.3
-Dgrounding.penalty.strategy=avg
```

#### Lenient (allow mixed attention, prefer some attended objects)
```bash
-Dgrounding.penalty.threshold=0.4
-Dgrounding.penalty.alpha=0.1
-Dgrounding.penalty.beta=0.2
-Dgrounding.penalty.strategy=max
```

## Technical Details

### Grounding Cache Dependency

The penalty computation relies on **cached binding lists** in `SingleClauseNode.cachedBindingLists`. These are populated during clause evaluation when:
- `cacheBLs = true` (set internally)
- The clause body is being evaluated for coverage

If bindings aren't cached, those groundings are skipped (assumes average weight, neutral impact).

### Performance Impact

- **Minor overhead**: Only computes penalty if binding lists are cached
- **Scales with**: Number of groundings × number of predicates per grounding
- Typical overhead: ~5-10% additional clause evaluation time

### Integration with Existing Weights

This penalty is **additive** to existing penalty terms:
```java
total_penalty = length_penalty + singleton_vars_penalty + grounding_penalty
score = variance_fit + total_penalty  // (then negated for maximization)
```

It does **not** replace the per-example attention weights used during variance computation.

## Debugging

Enable verbose output:
```bash
# Set debugLevel in ScoreRegressionNode.java:
protected final static int debugLevel = 2;  // 0=none, 1=some, 2=detailed

# Then rebuild
cd rdnboost
mvn clean package
```

Output will show:
```
%     Grounding penalty = 0.234 for clause: action(S) :- nearby(S,diver), oxygen_low(S).
%       Grounding analysis: k_high=45, k_low=12 => penalty=0.234
```

## Examples

### Example 1: Simple Clause
**Clause**: `fire(S) :- nearby(S, enemy)`

**Groundings** (3 examples):
- `nearby(state1, fish1)` → weight = 0.8 (≥ 0.5) → k_high++
- `nearby(state2, fish2)` → weight = 0.3 (< 0.5) → k_low++
- `nearby(state3, shark1)` → weight = 0.9 (≥ 0.5) → k_high++

**Penalty**: `-0.1 * 2 + 0.5 * 1 = -0.2 + 0.5 = 0.3`

### Example 2: Multi-Predicate Clause with `min` strategy
**Clause**: `fire(S) :- nearby(S, enemy), oxygen_low(S)`

**Groundings** (2 examples):
- Example 1:
  - `nearby(state1, fish1)` → weight = 0.8
  - `oxygen_low(state1)` → weight = 0.9
  - Aggregated (min) = 0.8 (≥ 0.5) → k_high++
- Example 2:
  - `nearby(state2, fish2)` → weight = 0.7
  - `oxygen_low(state2)` → weight = 0.3
  - Aggregated (min) = 0.3 (< 0.5) → k_low++

**Penalty**: `-0.1 * 1 + 0.5 * 1 = 0.4`

### Example 3: Multi-Predicate Clause with `avg` strategy
Same as Example 2, but using `avg`:
- Example 1: avg(0.8, 0.9) = 0.85 (≥ 0.5) → k_high++
- Example 2: avg(0.7, 0.3) = 0.5 (≥ 0.5) → k_high++

**Penalty**: `-0.1 * 2 + 0.5 * 0 = -0.2` (reward!)

## Rebuilding

After making changes:
```bash
cd rdnboost
mvn clean package
```

The JAR will be at: `rdnboost/target/boostsrl-weights-2.0.0.jar`

## Related Files

- `ScoreRegressionNode.java` - Main scoring logic with grounding penalty
- `FactWeightLoader.java` - Loads and manages per-predicate weights
- `WILLSetup.java` - Configuration and initialization
- `SingleClauseNode.java` - Clause node with grounding computation

## Future Enhancements

Possible extensions:
1. **Tiered penalties**: Different penalty values for very-low vs low attention
2. **Proportion thresholds**: Require X% of groundings to be attended
3. **Entropy-based**: Penalize high variance in grounding weights
4. **Branch-specific**: Different penalties for true/false branches
