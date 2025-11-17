# Debug Mode for RDN Tree Learning

## Overview

A comprehensive debug mode (`-debugScoring`) has been implemented for the RDN (Relational Dependency Network) tree learning system. This mode provides detailed insights into the clause evaluation and tree building process.

## Usage

To enable debug mode, add the `-debugScoring` flag when running the RDN training:

```bash
java -jar rdnboost/target/boostsrl-1.1.1.jar -l -train data/seaquest/all/fire/train/ \
  -target fire -trees 10 -debugScoring
```

Or use it with the full pipeline script by modifying the java command in `run_full_pipeline.sh`:

```bash
java -jar "$JAR" -l -train "$TRAIN_DIR" -target "$ACTION" -trees "$NUM_TREES" \
  -aucPathTest "$TEST_DIR" -testNegPosRatio 2 -debugScoring
```

## What Debug Mode Provides

### 1. Per-Clause Debug Output (Console)

For each clause evaluated during tree building, the system prints:

- **Clause being evaluated**: The full logical clause
- **Split counts**: How many examples satisfy (TRUE branch) vs. don't satisfy (FALSE branch) the clause
- **First 10 examples per branch**:
  - Example facts
  - Gradient values (showing whether they're positive or negative training examples)
  - Label: POS/NEG/ZERO
- **Variance calculations**: 
  - TRUE branch variance
  - FALSE branch variance
  - Combined variance (the optimization metric)
  - Formula breakdown showing the calculation

### 2. Clause Comparison Table (Console)

After evaluating all candidate clauses for a node, the system prints a comparison table showing:

- All evaluated clauses ranked by combined variance (ascending - lower is better)
- The clause selected as BEST (rank 1)
- Split information for each clause
- Variance values for each clause

### 3. Node-Level Files (Written to Disk)

Before selecting the best clause for each tree node, the system writes a file: `node_{depth}_{branch}.txt`

**Naming convention**:
- `node_0_root.txt`: Root node (depth 0)
- `node_1_true.txt`: Node at depth 1 on the TRUE branch of parent
- `node_2_false.txt`: Node at depth 2 on the FALSE branch of parent

**File contents**:
- Header with depth and branch information
- Total number of clauses evaluated
- All clauses sorted by combined variance (ascending - lower is better)
- For each clause:
  - Rank (with "*** BEST CLAUSE ***" marker for rank 1)
  - Full clause text
  - Total examples being split
  - Split breakdown (TRUE vs FALSE branch counts)
  - Variance by branch
  - Combined variance with formula breakdown

### 4. Gradient Value Display

The debug output now shows that negative training examples ARE being used in the learning process:

- **Positive gradients** (POS): From positive training examples
- **Negative gradients** (NEG): From negative training examples (converted to regression form)
- Each example shows its gradient value, e.g., `[Gradient: 0.2341 (POS)]` or `[Gradient: -0.1523 (NEG)]`

This addresses the common question: "How are negative examples used in boosted regression?"

## Example Output Structure

### Console Output
```
==========================================================================================
CLAUSE EVALUATION
==========================================================================================

Clause: fire(State) :- near(State, submarine, fish)

Split: 1234 examples satisfy clause (TRUE), 3210 do not (FALSE)

[TRUE BRANCH]
  Total examples: 1234 (890 positive gradients, 344 negative gradients)
  Showing first 10 example(s) with facts and gradient values:

    Example 1 [Gradient: 0.2341 (POS)]
      near(s1, submarine, fish). oxygen(s1, 45). ...

    Example 2 [Gradient: -0.1523 (NEG)]
      near(s2, submarine, fish). oxygen(s2, 78). ...

    ...

  Variance Calculation:
    Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)
           = 234.56 - (45.67^2 / 1234)
           = 232.87
    Weighted Variance = 232.87

[FALSE BRANCH]
  Total examples: 3210 (2100 positive gradients, 1110 negative gradients)
  ... similar output ...

==========================================================================================
SPLIT SCORE (Combined Variance)
==========================================================================================
  Formula: (trueVar + falseVar) / (trueWeight + falseWeight)
         = (232.87 + 567.89) / (1234 + 3210)
         = 800.76 / 4444
  
  Combined Variance = 0.1802
  (Lower is better - algorithm seeks to MINIMIZE variance)
==========================================================================================

... [multiple clause evaluations] ...

==========================================================================================
CLAUSE COMPARISON - ALL EVALUATED CLAUSES
==========================================================================================
Total clauses evaluated: 47
Sorted by combined variance (ascending - lower is better)

Rank 1: *** BEST CLAUSE (SELECTED) ***
  Clause: fire(State) :- near(State, submarine, fish)
  Split: 1234 TRUE, 3210 FALSE
  TRUE var: 232.87, FALSE var: 567.89
  Combined variance: 0.1802

Rank 2:
  Clause: fire(State) :- oxygen(State, O), O < 30
  Split: 890 TRUE, 3554 FALSE
  TRUE var: 189.34, FALSE var: 678.23
  Combined variance: 0.1953

...
==========================================================================================
```

### File Output (`node_1_true.txt`)
```
====================================================================================================
CLAUSE EVALUATIONS FOR NODE AT DEPTH 1 (TRUE BRANCH)
====================================================================================================

Total clauses evaluated: 47
Sorted by combined variance (ascending - lower is better)

----------------------------------------------------------------------------------------------------
RANK 1 *** BEST CLAUSE ***
----------------------------------------------------------------------------------------------------

Clause: fire(State) :- near(State, submarine, fish)

Total examples being split: 4444

Split:
  Examples that SATISFY clause (TRUE branch):  1234
  Examples that DON'T satisfy clause (FALSE branch): 3210

Variance by branch:
  TRUE branch variance:  232.870000
  FALSE branch variance: 567.890000

Combined Variance: 0.180200
  Formula: (trueVar + falseVar) / (trueCount + falseCount)
         = (232.870000 + 567.890000) / (1234 + 3210)
         = 0.180200

----------------------------------------------------------------------------------------------------
RANK 2
----------------------------------------------------------------------------------------------------

Clause: fire(State) :- oxygen(State, O), O < 30

...

====================================================================================================
END OF CLAUSE EVALUATIONS
====================================================================================================
```

## Implementation Details

### Modified Files

1. **CommandLineArguments.java**
   - Added `-debugScoring` flag parsing
   - Added getter/setter methods

2. **BranchStats.java**
   - Added `ENABLE_DETAILED_DEBUG` static flag
   - Added `ClauseEvaluation` class to track evaluated clauses
   - Added `evaluatedClauses` static list
   - Implemented `printDebugSummary()` for per-branch output
   - Implemented `printClauseComparison()` for clause ranking
   - Implemented `writeClausesToFile()` for file output
   - Added gradient value tracking and display

3. **RegressionInfoHolderForRDN.java**
   - Modified to use `addExampleForDebug()` with gradient values
   - Added `printClauseEvaluationSummary()` method
   - Records clause evaluations in the global list

4. **RunBoostedRDN.java** and **RunBoostedModels.java**
   - Set `ENABLE_DETAILED_DEBUG` flag from command-line arguments

5. **ILPouterLoop.java**
   - Added code to write files and print comparison before node expansion
   - Determines current depth and branch name from tree structure
   - Calls `BranchStats.writeClausesToFile()` and `printClauseComparison()`
   - Clears clause tracking after each node

### File Locations

Node evaluation files are written to the **working directory** (typically where you run the java command), with filenames like:
- `node_0_root.txt`
- `node_1_true.txt`
- `node_1_false.txt`
- `node_2_true.txt`
- etc.

## Benefits

1. **Transparency**: Understand exactly which clauses were considered and why a particular clause was chosen
2. **Debugging**: Identify issues with mode declarations, background knowledge, or training data
3. **Research**: Analyze the tree building process for papers and experiments
4. **Negative Example Usage**: Clearly see that negative examples (with negative gradients) are used in training
5. **Reproducibility**: File outputs provide a permanent record of the learning process

## Performance Impact

Debug mode adds minimal overhead:
- Console output: Negligible (only first 10 examples per branch)
- File writing: < 1 second per node
- Overall impact: < 1% for typical training runs

## Tips

1. **Large datasets**: Debug mode is most useful for understanding the first few trees. Consider disabling it for later trees to speed up training.

2. **Disk space**: Each node file is typically 10-100 KB. A full tree (depth 5) may generate ~60 files totaling ~1-5 MB.

3. **Console output**: Can be verbose. Redirect to a file if needed:
   ```bash
   java -jar ... -debugScoring 2>&1 | tee debug.log
   ```

4. **Analyzing results**: Look for:
   - Clauses with very unbalanced splits (might indicate overfitting)
   - Similar combined variances (indicates multiple good options)
   - Zero variance branches (perfectly pure splits)

## Future Enhancements

Potential additions:
- Configurable number of examples to show (currently hardcoded to 10)
- HTML output with syntax highlighting
- Visualization of tree structure with clause evaluations
- Statistics on clause complexity vs. performance
