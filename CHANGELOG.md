# Changelog

## Unified Pipeline (`run_full_pipeline.sh`)

### Features

1. **Combined Training and Evaluation**
   - Single script handles training, testing, calibration, and evaluation
   - Replaces separate `run.sh` and `run_train_inference_and_calibrate.sh`

2. **Automatic Weight File Management**
   - **Weighted training** (`weighted=true`): 
     - Ensures `train_pos_weights.txt` and `train_neg_weights.txt` are present
     - Restores files from `.bak` if previously moved
   
   - **Unweighted training** (`weighted=false`):
     - Moves weight files to `.bak` (backup) to ensure unweighted training
     - Prevents accidental weighted training when unweighted is requested

3. **Organized Output Structure**
   - All logs saved in model directory
   - Test inference: `{model_dir}/{action}/{action}_test_infer.log`
   - Train inference: `{model_dir}/{action}/{action}_train_infer.log`
   - Evaluation: `{model_dir}/eval_report.txt`

4. **Improved Evaluation Report**
   - Method 1: Direct argmax (non-calibrated)
   - Method 2: With logistic regression on the classifiers
   - Clear comparison showing which method performs better

### Usage

```bash
# Train and evaluate unweighted model
./run_full_pipeline.sh 3 10 false false

# Train and evaluate weighted model  
./run_full_pipeline.sh 3 10 true false

# Test only (skip training)
./run_full_pipeline.sh 3 10 false true
```

### File Structure

```
run_full_pipeline.sh          # Main unified pipeline
eval_calibrated.py            # Evaluation with calibration
check_calibration_progress.sh # Progress monitor

# Documentation
QUICKSTART.md                 # Quick reference
PIPELINE_README.md            # Detailed documentation
CALIBRATION_README.md         # Calibration explanation
```

### Migration from Old Scripts

**Old approach:**
```bash
# Step 1: Train and test
./run.sh 3 10 false false

# Step 2: Manual calibration setup
# (multiple manual steps)
```

**New approach:**
```bash
# Everything in one command
./run_full_pipeline.sh 3 10 false false
```

### Key Improvements Over Original

1. ✅ Automatic weight file management (no manual file moving)
2. ✅ Organized output (logs in model directories, not scattered)
3. ✅ Calibration included automatically
4. ✅ Single command execution
5. ✅ Clear naming ("with logistic regression" instead of "Platt-calibrated")
6. ✅ Better progress tracking

### Backward Compatibility

- Original `run.sh` still available if needed
- All existing model directories work with new scripts
- Weight files safely backed up (not deleted)
