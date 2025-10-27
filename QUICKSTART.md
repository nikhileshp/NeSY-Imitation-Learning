# Quick Start Guide

## Run Full Pipeline

Train, test, and evaluate models with calibration in one command:

```bash
./run_full_pipeline.sh 3 10 false false
```

Parameters: `<max_depth> <num_trees> <weighted> <only_test>`

## What You Get

After running, find all results in `rdn_models/seaquest/{model_name}/`:

- **`eval_report.txt`** - Performance comparison showing:
  - **Training Set**: How well the calibrator fits training data
    - Method 1: Direct argmax (baseline)
    - Method 2: With logistic regression (calibrated)
  - **Test Set**: Generalization to unseen data
    - Method 1: Direct argmax (baseline)
    - Method 2: With logistic regression (calibrated)
  
- **`{action}/{action}_test_infer.log`** - Test inference logs for each action

- **`{action}/{action}_train_infer.log`** - Training inference logs for each action

## Key Improvements

**Logistic regression calibration** learns to combine your 6 independent action classifiers by:
1. Understanding inter-action relationships
2. Correcting systematic biases
3. Making probabilities comparable across classifiers

Expected improvement: 2-5% increase in weighted F1 score (varies by model quality)

## Timing

- Full pipeline: ~2-4 hours
- Test only (skip training): ~30-60 minutes

## Monitor Progress

```bash
./check_calibration_progress.sh
```

## Previous Approach

Your original `run.sh` is still available but the new unified pipeline is recommended for:
- Cleaner organization (logs in model directories)
- Automatic calibration
- Single command execution
