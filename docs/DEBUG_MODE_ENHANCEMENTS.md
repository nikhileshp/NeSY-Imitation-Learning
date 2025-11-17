# Enhanced Debug Mode - Gradient Counts and Variance Computation Details

## Overview

The debug mode (`-debugScoring`) has been enhanced to provide detailed information about gradient counts and variance computation formulas in the node evaluation files.

## What's New

### 1. Gradient Count Breakdown

Each branch (TRUE and FALSE) now shows:
- **Total examples** in the branch
- **Number of positive gradients** (from positive training examples)
- **Number of negative gradients** (from negative training examples)

This clearly demonstrates that **both positive AND negative training examples are used** in the learning process.

### 2. Detailed Variance Computation

For each branch, the file now shows the complete variance calculation:
```
TRUE branch variance:  0.123456
  Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)
         = 234.567890 - (45.678901^2 / 1234.567890)
         = 234.567890 - 1.689012
         = 232.878878
```

This shows the exact mathematical computation used to evaluate each clause.

## Enhanced File Output Format

### Example: `node_1_true.txt`

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
    - Positive gradients (from positive training examples): 890
    - Negative gradients (from negative training examples): 344
  Examples that DON'T satisfy clause (FALSE branch): 3210
    - Positive gradients (from positive training examples): 2100
    - Negative gradients (from negative training examples): 1110

Variance by branch:

  TRUE branch variance:  232.870000
    Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)
           = 345.678901 - (123.456789^2 / 987.654321)
           = 345.678901 - 112.808901
           = 232.870000

  FALSE branch variance: 567.890000
    Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)
           = 789.012345 - (234.567890^2 / 876.543210)
           = 789.012345 - 221.122345
           = 567.890000

Combined Variance: 0.180200
  Formula: (trueVar + falseVar) / (trueCount + falseCount)
         = (232.870000 + 567.890000) / (1234 + 3210)
         = 0.180200

----------------------------------------------------------------------------------------------------
RANK 2
----------------------------------------------------------------------------------------------------

Clause: fire(State) :- oxygen(State, O), O < 30

... [similar format for other clauses]

====================================================================================================
END OF CLAUSE EVALUATIONS
====================================================================================================
```

## Implementation Details

### Modified Files

1. **BranchStats.java**
   - Enhanced `ClauseEvaluation` class to store:
     - Gradient counts: `truePosGradients`, `trueNegGradients`, `falsePosGradients`, `falseNegGradients`
     - Variance computation components: `trueSumOutputSquared`, `trueSumOutputAndNumGrounding`, `trueSumNumGroundingSquared`
     - Same for FALSE branch
   - Added `getGradientCounts()` method to count positive/negative gradients
   - Enhanced `writeClausesToFile()` to output:
     - Gradient count breakdown for each branch
     - Step-by-step variance formula evaluation for each branch

2. **RegressionInfoHolderForRDN.java**
   - Updated `ClauseEvaluation` creation to include:
     - Gradient counts from both branches
     - All variance computation components

### Variance Formula Explained

The variance for each branch is computed as:

```
variance = sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)
```

Where:
- `sumOfOutputSquared`: Sum of squared gradient values (outputs) for all examples in the branch
- `sumOfOutputAndNumGrounding`: Sum of (gradient × numGroundings) for all examples  
- `sumOfNumGroundingSquared`: Sum of (numGroundings²) for all examples

For most cases, `numGroundings = 1`, so this simplifies to:
```
variance = Σ(output²) - (Σ(output))² / n
```

Which is the standard variance formula: `E[X²] - E[X]²`

### Why This Matters

1. **Transparency**: You can now see exactly how the variance is computed for each clause
2. **Debugging**: If results seem unexpected, you can trace the exact numbers used
3. **Understanding Negative Examples**: The gradient counts prove that negative training examples (with negative gradients) contribute to the learning
4. **Research**: Provides detailed data for analyzing the tree building process

## Usage

No changes to usage - simply run with `-debugScoring`:

```bash
java -jar rdnboost/target/boostsrl-1.1.1.jar -l -train data/seaquest/all/fire/train/ \
  -target fire -trees 10 -debugScoring
```

The enhanced output will appear automatically in the `node_*.txt` files.

## Understanding the Output

### Gradient Counts

- **Positive gradients**: Come from positive training examples (y=1)
  - These have output/gradient values > 0
  - Indicate examples the model should predict as positive
  
- **Negative gradients**: Come from negative training examples (y=0)
  - These have output/gradient values < 0
  - Indicate examples the model should predict as negative

### Variance Interpretation

- **Lower variance = better split**: The clause that minimizes combined variance is chosen
- **TRUE vs FALSE variance**: Both are important - we want to reduce variance in both branches
- **Combined variance**: Weighted average of both branch variances

### Example Interpretation

If a clause shows:
```
TRUE branch:  1000 examples (800 pos gradients, 200 neg gradients), variance = 100.5
FALSE branch: 3000 examples (1500 pos gradients, 1500 neg gradients), variance = 200.3
Combined variance: 175.2
```

This means:
- The clause separates examples into 1000 (TRUE) and 3000 (FALSE)
- Both branches have a mix of positive and negative training examples
- The TRUE branch is more "pure" (800/1000 = 80% positive)
- The FALSE branch is balanced (50/50 split)
- Combined, this clause achieves a variance of 175.2

## Comparison with Previous Version

**Before**: Files showed only:
- Example counts per branch
- Variance values (no formula)
- No gradient information

**Now**: Files show:
- Example counts per branch
- **Positive/negative gradient breakdown**
- Variance values
- **Complete variance computation formula with actual values**
- Step-by-step calculation

This makes the debug output much more informative for understanding and debugging the tree learning process.
