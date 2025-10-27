# Calibration of Action Classifiers

## Problem
You have 6 independent binary classifiers (one per action: noop, fire, up, right, left, down). When combining them to make action predictions, simply taking `argmax` of raw probabilities is suboptimal because:

1. **Probabilities are not comparable**: Each classifier outputs its own probability scale
2. **No inter-action dependencies**: Independent classifiers can't learn trade-offs between actions
3. **Uncalibrated outputs**: Raw scores may not reflect true probabilities

## Solution: Platt Scaling (Logistic Calibration)

We use the training set to learn a calibration function that maps raw classifier outputs to calibrated probabilities that are comparable across actions.

### Process

#### Step 1: Generate Training Predictions
Run inference on the training data to get model predictions:
```bash
./run_train_inference_and_calibrate.sh <max_depth> <num_trees> <weighted>
```

This script:
1. Creates `train_infer` directories for each action
2. Generates `query_{action}.db` files containing all training examples
3. Runs inference using trained RDN models
4. Collects probabilities in `AUC/aucTemp.txt`

#### Step 2: Train Calibration Model
Using the training predictions and true labels from `train.csv`, fit a logistic regression model:
- **Input**: 6D vector of raw probabilities (one per action classifier)
- **Output**: Calibrated action prediction

The calibrator learns:
```python
calibrator = LogisticRegression()
calibrator.fit(X_train_scaled, y_train)
```

Where:
- `X_train`: Raw probabilities from each action classifier [batch_size, 6]
- `y_train`: True action labels [batch_size]

#### Step 3: Apply to Test Set
Transform test probabilities and make predictions:
```python
calibrated_probs = calibrator.predict_proba(X_test_scaled)
predictions = np.argmax(calibrated_probs, axis=1)
```

## Files Created

### Scripts
- `run_train_inference_and_calibrate.sh`: Main pipeline script
- `eval_calibrated.py`: Calibration training and evaluation

### Data Structure
```
data/seaquest/all/{action}/train/
├── train_facts.txt
├── train_pos.txt
├── train_neg.txt
└── train_infer/              # Created by pipeline
    ├── facts.txt
    ├── pos.txt
    ├── neg.txt
    ├── bk.txt
    ├── query_{action}.db     # All training queries
    └── AUC/
        └── aucTemp.txt       # Model predictions
```

## Usage

### Run Full Pipeline
```bash
# For unweighted model with depth 3, 10 trees
./run_train_inference_and_calibrate.sh 3 10 false

# For weighted model
./run_train_inference_and_calibrate.sh 3 10 true
```

### Run Only Calibration (if training inference already done)
```bash
python eval_calibrated.py --model_dir rdn_models/seaquest/unweighted_negpos_2_trees_10_depth_3
```

## Expected Results

The calibrated model should:
- Provide better-calibrated probability estimates
- Potentially improve weighted F1 score
- Show which predictions changed due to calibration

Output includes:
- Classification reports for both methods
- Confusion matrices
- Percentage of predictions that differ
- Saved report in `{model_dir}/eval_report_calibrated.txt`

## Why This Works

1. **Learns inter-action relationships**: The calibrator sees all 6 probabilities together and learns which combinations correspond to which true actions
2. **Corrects systematic biases**: If one classifier is consistently over/under-confident, calibration corrects this
3. **Comparable scales**: Maps all raw scores to a common probability scale

## Limitations

- Requires training data inference (adds computation time)
- Only helps if classifiers are reasonably accurate to begin with
- Best alternative: Train a single multi-class RDN model directly
