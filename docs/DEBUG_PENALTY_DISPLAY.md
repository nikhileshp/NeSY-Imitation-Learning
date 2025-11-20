# Debug Mode Penalty Display Enhancements

## Overview

Enhanced the debug mode output to display penalty breakdowns during clause evaluation, including the grounding-based attention penalty. This allows users to see exactly how penalties (length, singleton variables, and grounding penalties) affect clause scoring when using the `-debugScoring` flag.

## Changes Made

### 1. Extended ClauseEvaluation Class (`BranchStats.java`)

Added penalty tracking fields to the `ClauseEvaluation` class:
- `totalPenalty`: Sum of all penalties applied to the clause
- `lengthPenalty`: Penalty for clause length and singleton variables combined
- `singletonPenalty`: (Reserved for future separation from length penalty)
- `groundingPenalty`: Penalty based on attention weights of clause groundings

**File**: `rdnboost/src/edu/wisc/cs/will/ILP/Regression/BranchStats.java`
- Lines 34-38: Added penalty fields
- Lines 43-45, 63-66: Updated constructor to accept penalty parameters

### 2. Enhanced ScoreRegressionNode (`ScoreRegressionNode.java`)

Modified the `scoreThisNode` method to:
- Separate penalty calculation into components (length/singleton vs. grounding)
- Display detailed penalty breakdown in debug output
- Store penalty values in the node for later retrieval

**File**: `rdnboost/src/edu/wisc/cs/will/ILP/ScoreRegressionNode.java`
- Lines 84-121: Enhanced scoring with penalty breakdown
- Lines 100-110: Debug output showing penalty components

**File**: `rdnboost/src/edu/wisc/cs/will/ILP/SingleClauseNode.java`
- Lines 60-62: Added penalty tracking fields

### 3. Updated ClauseEvaluation Tracking (`RegressionInfoHolderForRDN.java`)

Modified to capture penalty information from the scored node and pass it to ClauseEvaluation objects:

**File**: `rdnboost/src/edu/wisc/cs/will/ILP/Regression/RegressionInfoHolderForRDN.java`
- Lines 21-22: Added caller node reference
- Lines 117-118: Store caller node reference
- Lines 209-227: Extract and pass penalty information to ClauseEvaluation

### 4. Enhanced Debug File Output (`BranchStats.java`)

Updated `writeClausesToFile` to include penalty breakdown in node debug files:

**File**: `rdnboost/src/edu/wisc/cs/will/ILP/Regression/BranchStats.java`
- Lines 391-400: Added penalty section to node files

### 5. Enhanced Console Comparison Output (`BranchStats.java`)

Updated `printClauseComparison` to include penalty columns in the comparison table:

**File**: `rdnboost/src/edu/wisc/cs/will/ILP/Regression/BranchStats.java`
- Lines 443-445: Added penalty columns to header
- Lines 462-464: Include penalty values in each row
- Lines 481-488: Show penalty details for best clause

## Output Format

### Console Output During Scoring

When debug mode is enabled, each clause evaluation now shows:

```
%     Score = -0.123456 (regressionFit = 0.100000, totalPenalty = 0.023456)
%       Penalty breakdown:
%         Length/Singleton = 0.010000
%         Grounding        = 0.013456
%       for clause: action(State) :- near(State, submarine, fish)
```

### Clause Comparison Table

The console comparison table now includes penalty columns:

```
Rank | Split      | Clause (truncated)               | TRUE     | FALSE    | Variance     | TotalPen   | GroundPen
-----+------------+----------------------------------+----------+----------+--------------+------------+----------
   1 | 123/456    | action(State) :- near(State,...  |      123 |      456 |     0.123456 |   0.023000 |   0.013000 ***
   2 | 234/345    | action(State) :- oxygen(State... |      234 |      345 |     0.145678 |   0.025000 |   0.015000
```

### Node Debug Files

Each `node_*.txt` file now includes penalty information for each clause:

```
Penalties:
  Total penalty:           0.023000
    Length/Singleton:      0.010000
    Grounding penalty:     0.013000
```

## Usage

### Enable Debug Mode

```bash
# Training with debug mode
./run_single_t_54_RZ.sh 3 10 true
```

### Without Debug Mode

Penalties are still computed and used in scoring, but detailed output is not displayed:

```bash
# Training without debug mode
./run_single_t_54_RZ.sh 3 10 false
# or
./run_single_t_54_RZ.sh 3 10
```

## Verification

The grounding penalty is automatically enabled when:
1. `fact_weights.txt` file exists in the training directory
2. System properties are set for penalty parameters (threshold, alpha, beta, strategy)

The configuration is displayed at startup:
```
% Grounding penalty configured: threshold=0.7 alpha=0.1 beta=0.5 strategy=min
```

## Understanding Penalty Values

### Total Penalty
The sum of all penalties. Added to the regression fit score (higher is worse since score is negated).

### Length/Singleton Penalty
Standard ILP penalties for:
- Clause length (number of literals)
- Singleton variables (variables appearing only once)
- Repeated predicates

Scaled by `scalingPenalties` (default: 0.1).

### Grounding Penalty
Based on attention weights from eye-tracking data:
- **Formula**: `grounding_penalty = -alpha * k_high + beta * k_low`
- `k_high`: Number of groundings with attention weight ≥ threshold
- `k_low`: Number of groundings with attention weight < threshold
- **Negative values** indicate reward (groundings involve attended objects)
- **Positive values** indicate penalty (groundings involve unattended objects)

### Interpretation

- **Small penalties** (near 0.0): Clause is concise and involves attended objects
- **Moderate penalties** (0.01-0.05): Some unattended groundings or longer clause
- **Large penalties** (> 0.1): Many unattended groundings or complex clause

The grounding penalty directly influences which clauses are selected during tree learning, biasing the system toward clauses that involve objects the player was attending to.

## Benefits

1. **Transparency**: See exactly how penalties affect clause selection
2. **Debugging**: Identify if grounding penalty is working as expected
3. **Parameter Tuning**: Understand impact of threshold, alpha, beta values
4. **Verification**: Confirm that attention weights are being used correctly

## Related Documentation

- `GROUNDING_PENALTY_README.md`: Grounding penalty feature specification
- `DEBUG_MODE_README.md`: General debug mode documentation
- `RUN_SINGLE_T_54_RZ_USAGE.md`: Usage guide for single_t training script
