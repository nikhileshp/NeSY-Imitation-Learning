# Usage Guide: run_single_t_54_RZ.sh

## Overview

Training and testing script for the `single_t/54_RZ_2461867` dataset with grounding penalty parameters enabled.

## Features

- **Grounding Penalty**: Automatic penalty based on attention weights from eye-tracking
  - `threshold=0.7`: Only groundings with attention ≥ 0.7 count as "attended"
  - `alpha=0.1`: Reward coefficient for attended groundings
  - `beta=0.5`: Penalty coefficient for unattended groundings  
  - `strategy=min`: Conservative aggregation (all predicates must be attended)

- **Debug Mode**: Optional detailed output showing clause evaluation and scoring

- **Full Pipeline**: Training → Testing → Train Inference → Calibration → Evaluation

## Usage

```bash
cd /home/nikhilesh/Projects/NeSY-Imitation-Learning
./run_single_t_54_RZ.sh <max_depth> <num_trees> [debug_mode]
```

### Parameters

- `max_depth`: Maximum depth of RDN trees (typically 3-8)
- `num_trees`: Number of boosting trees (typically 10-20)
- `debug_mode`: (Optional) "true" or "false" (default: false)

### Examples

#### Basic training (no debug)
```bash
./run_single_t_54_RZ.sh 3 10
```

#### With debug mode enabled
```bash
./run_single_t_54_RZ.sh 3 10 true
```

#### Deeper trees, more boosting
```bash
./run_single_t_54_RZ.sh 5 20
```

## What the Script Does

### Step 1: Training
Trains 6 independent action classifiers (fire, up, down, left, right, noop) using:
- Per-example attention weights from `fact_weights.txt`
- Grounding-based penalty to prefer clauses with attended objects
- BoostSRL weighted training mode

### Step 2: Testing
Runs inference on test data for each action classifier to generate predictions.

### Step 3: Training Inference
Runs inference on training data (needed for calibration step).

### Step 4: Calibration & Evaluation
- Trains logistic regression to combine the 6 independent classifiers
- Evaluates both uncalibrated (Method 1) and calibrated (Method 2) predictions
- Generates evaluation report with accuracy metrics

## Output

Models and results are saved to:
```
rdn_models/seaquest/single_t/54_RZ_2461867/negpos_2_trees_<NUM>_depth_<DEPTH>_grounding_penalty/
```

### Key Output Files

- `eval_report.txt`: Final evaluation metrics
- `<action>/WILLtheories/action_learnedWILLregressionTrees.txt`: Learned logical rules
- `<action>/action_test_infer.log`: Test inference logs
- `<action>/action_train_infer.log`: Training inference logs
- `<action>/node_*.txt`: Debug files showing clause evaluations (if debug mode enabled)

## Debug Mode Details

When debug mode is enabled (`debug_mode=true`), you'll see:

### Console Output
- Detailed clause evaluation for each candidate split
- Example facts and gradient values
- Variance calculations with formulas
- Clause comparison tables

### Node Files (`node_*.txt`)
- Written to model directory during training
- Shows all candidate clauses ranked by score
- Includes split statistics and variance values
- Format: `node_<depth>_<branch>.txt`

**Warning**: Debug mode generates extensive output and significantly slows training. Recommended only for:
- Small datasets or single trees
- Debugging specific issues
- Understanding the learning process

Redirect output to file if using debug mode:
```bash
./run_single_t_54_RZ.sh 3 10 true > debug_output.log 2>&1
```

## Performance

Typical runtime (without debug mode):
- Training: 10-30 min per action (60-180 min total)
- Test inference: 2-5 min per action (12-30 min total)
- Train inference: 5-15 min per action (30-90 min total)
- Calibration: < 1 min
- **Total**: 2-4 hours

Debug mode can increase training time by 5-10x.

## Modifying Grounding Penalty Parameters

To change the grounding penalty settings, edit these lines in the script:

```bash
# Lines 35-39
GROUNDING_THRESHOLD=0.7
GROUNDING_ALPHA=0.1
GROUNDING_BETA=0.5
GROUNDING_STRATEGY="min"
```

See `docs/GROUNDING_PENALTY_README.md` for parameter tuning guidelines.

## Troubleshooting

### JAR file not found
```bash
cd rdnboost
mvn clean package
```

### Out of memory errors
Edit the script and add heap size to Java commands:
```bash
java -Xmx8g -Dgrounding.penalty.threshold=... -jar "$JAR" ...
```

### Missing fact_weights.txt
The script expects `fact_weights.txt` in each action's training directory. If missing, the grounding penalty won't be applied.

### Calibration fails
Ensure both training and test inference completed successfully. Check for `AUC/aucTemp.txt` files in model directories.
