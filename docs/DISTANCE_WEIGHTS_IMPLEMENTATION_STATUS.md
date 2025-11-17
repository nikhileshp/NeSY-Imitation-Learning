# Distance Weights Feature - Implementation Status

## ✅ Completed (Phase 1 - Core Infrastructure)

### 1. Command-Line Flag ✅
**File**: `CommandLineArguments.java`
- Added `useDistanceWeights` flag constant
- Added `useDistanceWeightsFlag` private field
- Added parsing logic in `parseArgs()`
- Added `isUseDistanceWeights()` getter
- Added `setUseDistanceWeights()` setter

**Usage**: `-use-distance-weights` or `-use-distance-weights true/false`

### 2. FactWeightLoader Class ✅
**File**: `FactWeightLoader.java` (NEW)
- Loads weights from `fact_weights.txt`
- Parses format: `predicate(args). weight`
- Handles comments (%) and empty lines
- Normalizes fact strings (removes whitespace)
- Default weight = 1.0 for missing facts
- Validates weights (no negatives)
- Thread-safe HashMap for O(1) lookups

**Key Methods**:
- `loadWeights(String filePath)` - Load weights file
- `getWeight(String factString)` - Get weight for a fact
- `isWeightsLoaded()` - Check if loaded successfully
- `getWeightCount()` - Number of loaded weights

### 3. Compilation ✅
- All code compiles without errors
- Java 8 compatible

## 🔄 Remaining Work (Phase 2 - Integration)

### Step 3: Integrate FactWeightLoader into Setup

**Files to modify**:

#### `WILLSetup.java` or `LearnBoostedRDN.java`
```java
// Add field
private FactWeightLoader factWeightLoader = null;

// In setup method
if (cmdArgs.isUseDistanceWeights()) {
    factWeightLoader = new FactWeightLoader();
    String weightsPath = getTrainDirVal() + "/fact_weights.txt";
    factWeightLoader.loadWeights(weightsPath);
    
    // Pass to outer loop
    setup.getOuterLooper().setFactWeightLoader(factWeightLoader);
}
```

### Step 4: Add FactWeightLoader to ILPouterLoop

**File**: `ILPouterLoop.java`

```java
// Add field
private FactWeightLoader factWeightLoader = null;

// Add setter
public void setFactWeightLoader(FactWeightLoader loader) {
    this.factWeightLoader = loader;
}

// Add getter
public FactWeightLoader getFactWeightLoader() {
    return factWeightLoader;
}
```

### Step 5: Pass to SingleClauseNode During Scoring

**File**: `ScoreRegressionNode.java`

```java
public class ScoreRegressionNode extends ScoreSingleClauseByAccuracy {
    private FactWeightLoader weightLoader = null;
    private boolean useDistanceWeights = false;
    
    // Add constructor or setter
    public void setWeightLoader(FactWeightLoader loader) {
        this.weightLoader = loader;
        this.useDistanceWeights = (loader != null);
    }
    
    public double scoreThisNode(SearchNode nodeRaw) throws SearchInterrupted {
        SingleClauseNode node = (SingleClauseNode)nodeRaw;
        if (!Double.isNaN(node.score)) { return node.score; }
        
        double fit = (forMLNs ? node.regressionFitForMLNs() : node.regressionFit());
        double penalty = scalingPenalties * (getPenalties(node, true, true));
        
        // NEW: Apply distance weight multiplier
        if (useDistanceWeights && weightLoader != null) {
            double weightedAvg = computeWeightedAverage(node);
            fit = fit * weightedAvg;
        }
        
        double score = fit + penalty;
        node.score = -score;
        return -score;
    }
    
    private double computeWeightedAverage(SingleClauseNode node) {
        // This is the complex part - see detailed implementation below
        return 1.0; // Placeholder
    }
}
```

### Step 6: Implement computeWeightedAverage() - THE CORE LOGIC

This is the most complex part. Here's the full implementation:

```java
private double computeWeightedAverage(SingleClauseNode node) {
    try {
        // Get the clause
        Clause clause = node.getClause();
        if (clause == null) {
            return 1.0;
        }
        
        // Get examples that satisfy the clause (TRUE branch)
        LearnOneClause task = (LearnOneClause) node.task;
        List<Example> trueExamples = new ArrayList<>();
        
        // Collect examples that match this node
        for (Example ex : task.getPosExamples()) {
            if (node.matchesThisExample(ex, true)) {
                trueExamples.add(ex);
            }
        }
        
        if (trueExamples.isEmpty()) {
            return 1.0;
        }
        
        double totalWeight = 0.0;
        
        // For each example, compute sum of grounding weights
        for (Example ex : trueExamples) {
            double exampleWeight = computeClauseWeightForExample(clause, ex);
            totalWeight += exampleWeight;
        }
        
        // Return weighted average
        return totalWeight / trueExamples.size();
        
    } catch (Exception e) {
        Utils.println("% WARNING: Error computing weighted average: " + e.getMessage());
        return 1.0;
    }
}

private double computeClauseWeightForExample(Clause clause, Example ex) {
    double weight = 0.0;
    
    // Get literals in the clause body
    List<Literal> bodyLiterals = clause.posLiterals;
    if (bodyLiterals == null || bodyLiterals.isEmpty()) {
        return 1.0;
    }
    
    // For each literal in the clause body
    for (Literal lit : bodyLiterals) {
        // Get all groundings of this literal with this example
        weight += getGroundingWeightsForLiteral(lit, ex);
    }
    
    return weight;
}

private double getGroundingWeightsForLiteral(Literal lit, Example ex) {
    // This requires unification/grounding logic
    // For now, simplified version:
    
    // Convert literal to string representation
    String litString = lit.toString();
    
    // Try to get weight (this is simplified - real implementation
    // would need to enumerate all groundings)
    double weight = weightLoader.getWeight(litString);
    
    // If weight found, return it; otherwise return 1.0
    return (weight != 1.0) ? weight : 1.0;
}
```

### Step 7: Connect Setup to Scorer

**File**: Need to find where ScoreRegressionNode is instantiated

Typical location in `WILLSetup.java` or `LearnOneClause.java`:

```java
// When creating the scorer
ScoreRegressionNode scorer = new ScoreRegressionNode();

// If distance weights enabled, set the loader
if (cmdArgs.isUseDistanceWeights() && factWeightLoader != null) {
    scorer.setWeightLoader(factWeightLoader);
}

// Pass scorer to inner loop
innerLoop.setScorer(scorer);
```

## 📋 Testing Checklist

### Test 1: Basic Functionality
```bash
# Without flag (backward compatibility)
java -jar target/boostsrl-1.1.1.jar -l -train data/train/ -target action -trees 1

# Should work exactly as before ✓
```

### Test 2: With Flag, No Weights File
```bash
java -jar target/boostsrl-1.1.1.jar -l -train data/train/ -target action -trees 1 -use-distance-weights

# Should warn about missing fact_weights.txt
# Should use default weight 1.0
```

### Test 3: With Flag and Weights File
```bash
# Create sample fact_weights.txt
echo "nearby(state1,fish1). 0.98" > data/train/fact_weights.txt
echo "nearby(state1,fish2). 0.84" >> data/train/fact_weights.txt

java -jar target/boostsrl-1.1.1.jar -l -train data/train/ -target action -trees 1 -use-distance-weights

# Should load weights
# Should use weighted scoring
```

### Test 4: With Debug Output
```bash
java -jar target/boostsrl-1.1.1.jar -l -train data/train/ -target action -trees 1 \
  -use-distance-weights -debugScoring

# Should show weighted average calculations
```

## 🐛 Known Challenges

### Challenge 1: Grounding Enumeration
The hardest part is enumerating all groundings of a literal with an example. This requires:
- Unification between literal and facts
- Accessing the knowledge base/facts
- Handling variables in the literal

**Solution**: May need to access `BindingList` and use theorem prover to find all groundings.

### Challenge 2: Performance
Computing weighted average for every clause could be expensive.

**Solution**: 
- Cache results per (clause, example) pair
- Only compute for top-K clauses
- Use lazy evaluation

### Challenge 3: Access to Facts
Need access to ground facts to enumerate groundings.

**Solution**: Access via `task.getContext()` or `stringHandler.getClausebase()`

## 📝 Alternative Simplified Implementation

If grounding enumeration is too complex, use a **simplified approach**:

### Simplified Approach: Predicate-Level Weights

Instead of per-grounding weights, use per-predicate average weights:

```java
// In FactWeightLoader, add:
private Map<String, List<Double>> predicateWeights;  // predicate -> list of weights

// Compute average weight per predicate
public double getAverageWeightForPredicate(String predicateName) {
    List<Double> weights = predicateWeights.get(predicateName);
    if (weights == null || weights.isEmpty()) {
        return 1.0;
    }
    double sum = 0.0;
    for (double w : weights) sum += w;
    return sum / weights.size();
}

// In computeWeightedAverage():
private double computeWeightedAverage(SingleClauseNode node) {
    Clause clause = node.getClause();
    List<Literal> bodyLiterals = clause.posLiterals;
    
    double totalWeight = 0.0;
    int literalCount = 0;
    
    for (Literal lit : bodyLiterals) {
        String predName = lit.predicateName.name;
        totalWeight += weightLoader.getAverageWeightForPredicate(predName);
        literalCount++;
    }
    
    return (literalCount > 0) ? totalWeight / literalCount : 1.0;
}
```

This is **much simpler** and still provides benefit: clauses using predicates with higher average weights (i.e., closer objects on average) will be preferred.

## 🎯 Next Steps

1. Choose implementation approach (full vs simplified)
2. Implement `computeWeightedAverage()` method
3. Connect FactWeightLoader to training pipeline
4. Test with sample data
5. Add debug output for weighted average
6. Document usage in README

## 📂 Files Modified Summary

**Completed**:
- ✅ `CommandLineArguments.java` - Added flag
- ✅ `FactWeightLoader.java` - Created new class

**Remaining**:
- ⏳ `WILLSetup.java` - Load weights at startup
- ⏳ `ILPouterLoop.java` - Pass loader to components
- ⏳ `ScoreRegressionNode.java` - Implement weighted scoring
- ⏳ `LearnOneClause.java` - Connect scorer to loader

**Estimated effort**: 2-4 hours for full implementation

The core infrastructure is complete. The remaining work is integrating the loader into the scoring pipeline and implementing the weighted average calculation.
