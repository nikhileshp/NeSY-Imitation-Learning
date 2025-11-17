# Distance-Based Grounding Weights Specification

## Overview

Add support for per-grounding distance weights that affect clause scoring. The score becomes:
```
final_score = combined_variance × weighted_average_multiplier
```

## Feature Components

### 1. Command-Line Flag

**Flag**: `--use-distance-weights` (boolean, optional)

**Usage**:
```bash
java -jar boostsrl.jar -l -train data/train/ -target action \
  -trees 10 -use-distance-weights
```

**Default**: `false` (disabled)

### 2. Fact Weights File Format

**File Location**: `<train_dir>/fact_weights.txt`

**Format**:
```
predicate_name(arg1, arg2, ...). weight
```

**Example** (`fact_weights.txt`):
```
nearby(state1, fish1). 0.98
nearby(state1, fish2). 0.84
nearby(state1, fish3). 0.25
nearby(state2, fish1). 1.00
near(state1, diver1). 0.50
oxygen_low(state1). 1.00
```

**Parsing Rules**:
- One fact per line
- Fact followed by a period, then weight (space-separated)
- Lines without weights are ignored (weight = 1.0 default)
- Comments start with `%`

### 3. Weighted Average Multiplier Calculation

#### Per-Clause Computation

For each clause being scored:

**Step 1**: For each example, compute clause weight
```
For example i:
  clause_weight_i = 0
  For each predicate p in clause body:
    For each grounding g of p with example i:
      clause_weight_i += grounding_weight(p, g)
```

**Step 2**: Compute weighted average across all examples
```
weighted_avg = sum(clause_weight_i) / num_examples
```

Where `num_examples` = total examples considered for scoring the clause

#### Example Calculation

**Clause**: `action(S) :- nearby(S, Fish), oxygen_low(S)`

**Example 1** (3 fish nearby, low oxygen):
```
Groundings:
  nearby(state1, fish1)  weight = 0.98
  nearby(state1, fish2)  weight = 0.84
  nearby(state1, fish3)  weight = 0.25
  oxygen_low(state1)     weight = 1.00

clause_weight_1 = 0.98 + 0.84 + 0.25 + 1.00 = 3.07
```

**Example 2** (1 fish nearby, low oxygen):
```
Groundings:
  nearby(state2, fish1)  weight = 1.00
  oxygen_low(state2)     weight = 1.00

clause_weight_2 = 1.00 + 1.00 = 2.00
```

**Example 3** (no fish nearby, low oxygen - FALSE branch):
```
clause_weight_3 = 0  (doesn't satisfy clause)
```

**Weighted Average** (for TRUE branch with 2 examples):
```
weighted_avg = (3.07 + 2.00) / 2 = 2.535
```

### 4. Modified Scoring Formula

#### Current Formula
```java
variance = sumOfOutputSquared - (sumOfOutputAndNumGrounding²) / sumOfNumGroundingSquared
```

#### New Formula (when `--use-distance-weights` enabled)
```java
// Step 1: Compute standard variance (unchanged)
variance = sumOfOutputSquared - (sumOfOutputAndNumGrounding²) / sumOfNumGroundingSquared

// Step 2: Compute weighted average multiplier
weighted_avg = computeWeightedAverageForClause(clause, examples)

// Step 3: Final score
final_score = variance × weighted_avg
```

### 5. Implementation Files to Modify

#### CommandLineArguments.java
```java
// Add field
public static final String useDistanceWeights = "use-distance-weights";
private boolean useDistanceWeightsFlag = false;

// Add parsing in parseArgs()
if (argMatches(args[i], useDistanceWeights)) {
    useDistanceWeightsFlag = true;
    continue;
}

// Add getter
public boolean isUseDistanceWeights() {
    return useDistanceWeightsFlag;
}
```

#### New Class: FactWeightLoader.java
```java
package edu.wisc.cs.will.ILP.Regression;

import java.io.*;
import java.util.*;

public class FactWeightLoader {
    // Map: "predicate(arg1,arg2)" -> weight
    private Map<String, Double> factWeights = new HashMap<>();
    
    public void loadWeights(String filePath) {
        // Parse fact_weights.txt
        // Store in factWeights map
    }
    
    public double getWeight(String factString) {
        return factWeights.getOrDefault(factString, 1.0);
    }
    
    public double getClauseWeightForExample(Clause clause, Example example) {
        // For each literal in clause body
        // For each grounding of that literal with example
        // Sum up weights
        return totalWeight;
    }
}
```

#### ScoreRegressionNode.java
```java
public double scoreThisNode(SearchNode nodeRaw) throws SearchInterrupted {
    SingleClauseNode node = (SingleClauseNode)nodeRaw;
    
    double fit = node.regressionFit();
    double penalty = scalingPenalties * getPenalties(node, true, true);
    
    // NEW: If distance weights enabled
    if (useDistanceWeights) {
        double weightedAvg = computeWeightedAverage(node);
        fit = fit × weightedAvg;  // Multiply variance by weighted average
    }
    
    double score = fit + penalty;
    node.score = -score;
    return -score;
}

private double computeWeightedAverage(SingleClauseNode node) {
    // Get clause
    // Get examples that match
    // For each example, compute clause weight
    // Return average
}
```

#### SingleClauseNode.java
```java
// Add method to compute weighted average for this node
public double computeWeightedAverageMultiplier(FactWeightLoader weightLoader) {
    List<Example> trueExamples = getTrueBranchExamples();
    double totalWeight = 0.0;
    
    for (Example ex : trueExamples) {
        double clauseWeight = weightLoader.getClauseWeightForExample(
            this.getClause(), ex);
        totalWeight += clauseWeight;
    }
    
    return totalWeight / trueExamples.size();
}
```

### 6. Data Flow

```
1. User provides: --use-distance-weights flag

2. At startup:
   - Load fact_weights.txt into FactWeightLoader
   - Pass loader to scoring components

3. During clause evaluation:
   - Compute standard variance (unchanged)
   - If flag enabled:
     - For each example, compute clause weight
     - Compute weighted average
     - Multiply variance by weighted average
   
4. Clause selection:
   - Choose clause with best (lowest) final score
   - Score = variance × weighted_avg
```

### 7. Backward Compatibility

**When `--use-distance-weights` is NOT provided:**
- All existing behavior unchanged
- No `fact_weights.txt` file needed
- Standard variance-only scoring

**When flag IS provided but no `fact_weights.txt`:**
- Default all weights to 1.0
- Weighted average = number of groundings
- Equivalent to multiplying variance by grounding count

### 8. Example Scenarios

#### Scenario A: Close objects preferred

**Clause 1**: `action(S) :- nearby(S, Fish)`
- Example 1: 3 fish, weights [0.98, 0.84, 0.25], sum = 2.07
- Example 2: 1 fish, weight [1.00], sum = 1.00
- Weighted avg = (2.07 + 1.00) / 2 = 1.535
- If variance = 0.05, score = 0.05 × 1.535 = 0.0768

**Clause 2**: `action(S) :- near(S, Enemy)`
- Example 1: 3 enemies, weights [0.25, 0.25, 0.25], sum = 0.75
- Example 2: 1 enemy, weight [1.00], sum = 1.00
- Weighted avg = (0.75 + 1.00) / 2 = 0.875
- If variance = 0.05, score = 0.05 × 0.875 = 0.0438 ← BETTER (lower)

Result: Clause 2 preferred because objects are at better distances

#### Scenario B: High-weight objects important

**Clause 1**: Matches high-weight objects
- Weighted avg = 2.5
- Variance = 0.1
- Score = 0.1 × 2.5 = 0.25

**Clause 2**: Matches low-weight objects
- Weighted avg = 0.5
- Variance = 0.1
- Score = 0.1 × 0.5 = 0.05 ← BETTER

Result: Prefers clauses involving high-importance (close) objects

### 9. File Structure Example

```
data/seaquest/all/fire/train/
├── train_bk.txt         # Background knowledge
├── train_facts.txt      # Ground facts
├── fact_weights.txt     # NEW: Grounding weights
├── train_pos.txt        # Positive examples
└── train_neg.txt        # Negative examples
```

### 10. Testing Strategy

**Test 1**: Without flag (backward compatibility)
```bash
java -jar boostsrl.jar -l -train data/train/ -target action -trees 1
# Should work exactly as before
```

**Test 2**: With flag, no weights file
```bash
java -jar boostsrl.jar -l -train data/train/ -target action -trees 1 -use-distance-weights
# Should default all weights to 1.0
```

**Test 3**: With flag and weights file
```bash
# Create fact_weights.txt with sample weights
java -jar boostsrl.jar -l -train data/train/ -target action -trees 1 -use-distance-weights
# Should use weighted scoring
```

**Test 4**: Verify scoring
```bash
# Add -debugScoring to see how weights affect clause selection
java -jar boostsrl.jar -l -train data/train/ -target action -trees 1 \
  -use-distance-weights -debugScoring
```

### 11. Debug Output Enhancement

When both `-debugScoring` and `-use-distance-weights` are enabled, show:

```
===========================================================================
CLAUSE EVALUATION WITH DISTANCE WEIGHTS
===========================================================================

Clause: action(State) :- nearby(State, Fish)

TRUE Branch - Example Weights:
  Example 1: clause_weight = 2.07 (fish1:0.98 + fish2:0.84 + fish3:0.25)
  Example 2: clause_weight = 1.00 (fish1:1.00)
  ...
  Weighted Average = 1.535

Variance: 0.05000
Weighted Average Multiplier: 1.535
Final Score: 0.07675

===========================================================================
```

### 12. Implementation Priority

1. **Phase 1** (Core functionality):
   - Add command-line flag
   - Create FactWeightLoader
   - Modify scoring to use weighted average

2. **Phase 2** (Integration):
   - Connect loader to tree learning pipeline
   - Test with sample data

3. **Phase 3** (Enhancement):
   - Add debug output
   - Optimize performance
   - Add validation/error handling

### 13. Performance Considerations

- **File I/O**: Load weights once at startup, cache in memory
- **Lookup**: Use HashMap for O(1) weight lookups
- **Computation**: Only compute weighted average for top-scoring clauses if needed

### 14. Error Handling

- Missing `fact_weights.txt`: Warn and default to weight=1.0
- Malformed lines: Skip with warning
- Negative weights: Error and exit
- Missing predicates: Default to weight=1.0

## Summary

This feature allows the RDN system to prefer clauses that involve groundings with higher distance-based weights (i.e., closer objects), by multiplying the variance score by a weighted average multiplier computed from the grounding weights.
