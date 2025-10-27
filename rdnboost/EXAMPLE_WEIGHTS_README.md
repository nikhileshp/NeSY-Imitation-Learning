# Setting Custom Weights for Examples in RDNBoost

## Overview

The modified RDNBoost code now supports setting different weights for different training examples. This is useful for:
- Emphasizing important examples
- De-emphasizing noisy or uncertain examples
- Handling imbalanced datasets with fine-grained control
- Incorporating domain knowledge about example importance

## How It Works

The system automatically looks for weight files when loading training examples. These files specify custom weights for individual examples. Examples not listed in the weight files default to weight 1.0.

## Weight File Format

### File Naming Convention

Place weight files in the same directory as your example files:
- **Positive examples**: `{prefix}_pos_weights.txt`
- **Negative examples**: `{prefix}_neg_weights.txt`

For the ICML dataset:
- `sample/ICML/train/train_pos_weights.txt`
- `sample/ICML/train/train_neg_weights.txt`

### File Format

```
% Comments start with %
% Format: ExampleLiteral: weight

CoAuthor("Anish_Athalye","Nicholas_Carlini"): 2.5
CoAuthor("Judy_Hoffman","Trevor_Darrell"): 2.0
CoAuthor("Kevin_Kwok","Jessy_Lin"): 0.5
```

**Rules:**
1. Each line has format: `ExampleLiteral: weight`
2. Use the same literal format as in your `_pos.txt` and `_neg.txt` files
3. The colon `:` separates the example from the weight
4. Weights are floating-point numbers (e.g., 0.5, 1.0, 2.5, 3.0)
5. Examples not listed default to weight 1.0
6. Comments start with `%` or `//`
7. Empty lines are ignored

## Example Usage

### Step 1: Create Weight Files

Create `train_pos_weights.txt`:
```
% High-importance positive examples
CoAuthor("Pieter_Abbeel","Sergey_Levine"): 3.0
CoAuthor("Anish_Athalye","Nicholas_Carlini"): 2.5

% Low-importance positive examples  
CoAuthor("Kevin_Kwok","Jessy_Lin"): 0.5
```

Create `train_neg_weights.txt`:
```
% Important negative examples to learn from
CoAuthor("Suprovat_Ghoshal","Jason_D_Lee"): 1.5
CoAuthor("Jason_D_Lee","Aurick_Zhou"): 1.2
```

### Step 2: Run Training

The weights are automatically loaded when you run training:

```bash
java -jar BoostSRL.jar -l -train sample/ICML/train/ -target CoAuthor -trees 10
```

You should see output like:
```
% Loaded 8 custom example weights.
```

### Step 3: Verify

The weighted variance calculation in splits will use your custom weights:
```
weightedVariance = Σ(w_i * g_i²) - (Σ(w_i * g_i))² / Σ(w_i)
```

Where `w_i` is your custom weight (or 1.0 if not specified).

## Advanced Usage

### Matching Format

The system tries multiple formats to match examples:
- With spaces: `CoAuthor("A", "B")`
- Without spaces: `CoAuthor("A","B")`

Make sure your weight file format matches your example files.

### Weight Guidelines

- **Weight > 1.0**: Emphasize this example (higher influence on splits)
- **Weight = 1.0**: Normal weight (default)
- **Weight < 1.0**: De-emphasize this example (lower influence on splits)
- **Weight = 0.0**: Effectively ignore (not recommended, just remove from dataset instead)

### Practical Weight Ranges

- **High importance**: 2.0 - 5.0
- **Normal**: 0.8 - 1.2  
- **Low importance**: 0.3 - 0.7
- **Very low**: 0.1 - 0.3

Avoid extreme weights (e.g., > 100) as they can dominate learning.

## Implementation Details

The modification adds three methods to `WILLSetup.java`:

1. **`loadExampleWeightsIfAvailable()`**: Loads weight files if they exist
2. **`loadWeightsFromFile(filename)`**: Parses a weight file
3. **`applyExampleWeight(rex)`**: Applies weight to a `RegressionRDNExample`

Weights are loaded once per training run and applied when examples are converted to `RegressionRDNExample` objects in the `getJointExamples()` method.

## Testing

To test if weights are being applied:

1. Create a small weight file with a few examples
2. Run training with debug output
3. Check for the "Loaded N custom example weights" message
4. Examine split scores - heavily weighted examples should influence splits more

## Troubleshooting

**Weights not loading?**
- Check file naming: must be `{prefix}_pos_weights.txt` or `{prefix}_neg_weights.txt`
- Check file location: must be in same directory as `_pos.txt` and `_neg.txt`
- Check format: ensure colon `:` separates example from weight

**Weight mismatch warnings?**
- Example literals in weight file must exactly match those in example files
- Check for trailing periods (system handles both with and without)
- Check for spacing differences (system tries both formats)

**Examples still have weight 1.0?**
- Only examples listed in weight files get custom weights
- Others default to 1.0 (this is intentional)

## Related Parameters

You can still use global weight parameters for class imbalance:
```prolog
setParam: weightOnPosExamples = 1.0.
setParam: weightOnNegExamples = 0.5.
```

Per-example weights are multiplicative with these global weights.
