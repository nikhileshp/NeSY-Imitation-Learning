# Unified Training and Evaluation Pipeline

This pipeline combines training, testing, calibration, and evaluation into a single script.

## Usage

```bash
./run_full_pipeline.sh <max_depth> <num_trees> <weighted> <only_test>
```

### Parameters

- `max_depth`: Maximum tree depth (e.g., 3)
- `num_trees`: Number of trees (e.g., 10)
- `weighted`: Use weighted training? (`true` or `false`)
- `only_test`: Skip training and only test? (`true` or `false`)

### Examples

```bash
# Train and evaluate unweighted model
./run_full_pipeline.sh 3 10 false false

# Train and evaluate weighted model
./run_full_pipeline.sh 3 10 true false

# Only test existing model (skip training)
./run_full_pipeline.sh 3 10 false true
```

## Pipeline Steps

### Step 1: Training (optional)
Trains 6 RDN models (one per action: fire, up, down, left, right, noop) using BoostSRL.

**Weight File Handling:**
- **Weighted training** (`weighted=true`): Uses `train_pos_weights.txt` and `train_neg_weights.txt`
- **Unweighted training** (`weighted=false`): Moves weight files to `.bak` (backup) to ensure unweighted training
- Weight files are automatically restored when switching back to weighted training

**Skipped if:** `only_test=true`

### Step 2: Testing
Runs inference on test data for all 6 action models.

**Outputs:** 
- Test probabilities in `data/seaquest/all/{action}/test/AUC/aucTemp.txt`
- Inference logs in `{model_dir}/{action}/{action}_test_infer.log`

### Step 3: Training Inference (for Calibration)
Runs inference on training data to get model predictions for calibration.

**Outputs:**
- Training probabilities in `data/seaquest/all/{action}/train/train_infer/AUC/aucTemp.txt`
- Inference logs in `{model_dir}/{action}/{action}_train_infer.log`

### Step 4: Evaluation
Trains a logistic regression calibrator and evaluates both methods:
1. **Direct argmax**: Takes argmax of raw probabilities
2. **With logistic regression**: Calibrates probabilities using logistic regression trained on training data

**Outputs:**
- Evaluation report in `{model_dir}/eval_report.txt`

## Output Structure

```
rdn_models/seaquest/{model_name}/
├── eval_report.txt              # Evaluation results (both methods)
├── fire/
│   ├── fire_test_infer.log      # Test inference log
│   ├── fire_train_infer.log     # Train inference log (for calibration)
│   └── ...                       # Model files
├── up/
│   ├── up_test_infer.log
│   ├── up_train_infer.log
│   └── ...
├── down/
│   ├── down_test_infer.log
│   ├── down_train_infer.log
│   └── ...
├── left/
│   ├── left_test_infer.log
│   ├── left_train_infer.log
│   └── ...
├── right/
│   ├── right_test_infer.log
│   ├── right_train_infer.log
│   └── ...
└── noop/
    ├── noop_test_infer.log
    ├── noop_train_infer.log
    └── ...
```

## Evaluation Report Format

The `eval_report.txt` contains results for both training and test sets:

### Training Set Performance
1. **METHOD 1: Direct argmax (non-calibrated) - TRAINING**
   - Classification report (precision, recall, F1-score per action)
   - Confusion matrix
   
2. **METHOD 2: With logistic regression on the classifiers - TRAINING**
   - Classification report (calibrated predictions)
   - Confusion matrix
   - Comparison statistics

### Test Set Performance
1. **METHOD 1: Direct argmax (non-calibrated) - TEST**
   - Classification report (precision, recall, F1-score per action)
   - Confusion matrix
   
2. **METHOD 2: With logistic regression on the classifiers - TEST**
   - Classification report (calibrated predictions)
   - Confusion matrix
   - Comparison statistics

**Note:** Training performance shows how well the calibrator fits the training data, while test performance shows generalization to unseen data.

## Why Calibration?

You're training 6 independent binary classifiers (one per action). Their probability outputs are not directly comparable because:
- Each classifier learns on different data distributions
- They don't account for inter-action dependencies
- Raw scores may not be well-calibrated

The logistic regression calibrator:
- Sees all 6 probabilities together
- Learns which probability combinations correspond to which true actions
- Corrects systematic biases in individual classifiers
- Maps to a common probability scale

## Performance Notes

- **Training**: ~10-30 minutes per action (depends on data size and tree depth)
- **Test Inference**: ~2-5 minutes per action
- **Train Inference**: ~5-15 minutes per action (larger dataset)
- **Calibration**: < 1 minute

**Total time for full pipeline**: ~2-4 hours (for typical configurations)

## Monitoring Progress

Use the progress checker while the pipeline runs:

```bash
./check_calibration_progress.sh
```

This shows which actions have completed inference and how many predictions have been generated.
