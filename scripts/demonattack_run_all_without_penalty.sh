#!/bin/bash

set -e  # stop if any command fails

# Parse command line arguments
MAX_DEPTH=$1
NUM_TREES=$2
DEBUG_MODE=$3  # "true" or "false" (optional, default: false)
TEST_ONLY=$4   # "true" or "false" (optional, default: false)

if [ -z "$MAX_DEPTH" ] || [ -z "$NUM_TREES" ]; then
    echo "Usage: $0 <max_depth> <num_trees> [debug_mode] [test_only]"
    echo "Example: $0 3 10"
    echo "Example: $0 3 10 true  # Enable debug mode"
    echo "Example: $0 3 10 false true  # Test only (skip training)"
    exit 1
fi

# Default debug mode to false if not specified
if [ -z "$DEBUG_MODE" ]; then
    DEBUG_MODE="false"
fi

# Default test_only to false if not specified
if [ -z "$TEST_ONLY" ]; then
    TEST_ONLY="false"
fi

echo "=================================================="
echo "Training/Testing pipeline for DemonAttack"
echo "=================================================="
echo "Max tree depth: $MAX_DEPTH"
echo "Number of trees: $NUM_TREES"
echo "Debug mode: $DEBUG_MODE"
echo "Test only: $TEST_ONLY"
echo ""

# Configuration
JAR="rdnboost/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils"
NEG_POS_RATIO=2
DATA_BASE="data/demonattack/all"

# Seeds to test with (can be modified to test different random samples)
SEEDS=(1729 42 123 456 789)

# Base model directory (will be extended with seed info)
MODEL_BASE_PREFIX="rdn_models/demonattack/all/negpos_${NEG_POS_RATIO}_trees_${NUM_TREES}_depth_${MAX_DEPTH}"

# Debug flag for Java
if [ "$DEBUG_MODE" == "true" ]; then
    DEBUG_FLAG="-debugScoring"
    echo "Debug scoring enabled - verbose output will be generated"
else
    DEBUG_FLAG=""
fi

# DemonAttack actions to train/test (6 actions)
ACTIONS=("noop" "fire" "right" "left")

if [ "$TEST_ONLY" == "true" ]; then
    echo "=========================================="
    echo "TEST-ONLY MODE: Skipping training and BK update"
    echo "=========================================="
    echo ""
else
    # Update background knowledge with max_depth
    echo "Updating background knowledge files with max_depth=$MAX_DEPTH..."
    python src/change_bk.py --max_depth "$MAX_DEPTH" --base_dir "$DATA_BASE"

    # ============================================================================
    # STEP 1: TRAINING
    # ============================================================================
    echo ""
    echo "==========================================="
    echo "STEP 1: Training models"
    echo "==========================================="
    echo ""

    for action in "${ACTIONS[@]}"; do
        TRAIN_DIR="$DATA_BASE/$action/train"
        MODEL_DIR="$MODEL_BASE_PREFIX/$action"
        
        echo "Training $action..."
        mkdir -p "$MODEL_DIR"
        
        java -jar "$JAR" \
            -l \
            -train "$TRAIN_DIR" \
            -target "action" \
            -trees "$NUM_TREES" \
            -aucJarPath "$AUC_JAR" \
            -negPosRatio "$NEG_POS_RATIO" \
            -model "$MODEL_DIR" \
            $DEBUG_FLAG
        
        echo "✅ Completed training for $action"
        echo ""
    done
fi

# ============================================================================
# STEP 2: TESTING WITH MULTIPLE SEEDS
# ============================================================================
echo ""
echo "==========================================="
echo "STEP 2: Running inference on test data with multiple seeds"
echo "==========================================="
echo ""

# Use the same directory as training
echo "Testing model: $MODEL_BASE_PREFIX"
MODEL_DIRS=("${MODEL_BASE_PREFIX}")
echo "Testing configuration: $(basename $MODEL_BASE_PREFIX)"
echo ""

# Loop through each model directory
for MODEL_BASE in "${MODEL_DIRS[@]}"; do
    echo "=========================================="
    echo "Testing model: $(basename $MODEL_BASE)"
    echo "=========================================="
    
    # Loop through each seed
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- Testing with seed: $SEED ---"
        
        for action in "${ACTIONS[@]}"; do
            TEST_DIR="$DATA_BASE/$action/test"
            MODEL_DIR="$MODEL_BASE/$action"
            LOG_FILE="${MODEL_DIR}/action_test_infer_seed_${SEED}.log"
            
            # Check if model exists
            if [ ! -d "$MODEL_DIR/bRDNs" ]; then
                echo "⚠️  Skipping $action - model not found at $MODEL_DIR"
                continue
            fi
            
            mkdir -p "$MODEL_DIR"
            > "$LOG_FILE"
            
            echo "Testing $action with seed $SEED..."
            
            {
                echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test, Seed: $SEED)"
                
                java -jar "$JAR" \
                    -i \
                    -model "$MODEL_DIR" \
                    -test "$TEST_DIR" \
                    -target "action" \
                    -trees "$NUM_TREES" \
                    -testNegPosRatio "$NEG_POS_RATIO" \
                    -seed "$SEED" \
                    -aucJarPath "$AUC_JAR"
                
                echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test, Seed: $SEED)"
            } >> "$LOG_FILE" 2>&1
            
            echo "✅ Completed test inference for $action (seed $SEED)"
        done
    done
    
    echo ""
done

# ============================================================================
# STEP 3: TRAINING INFERENCE (for calibration)
# ============================================================================
echo ""
echo "==========================================="
echo "STEP 3: Running inference on training data (for calibration)"
echo "==========================================="
echo ""

for action in "${ACTIONS[@]}"; do
    TRAIN_DIR="$DATA_BASE/$action/train"
    MODEL_DIR="$MODEL_BASE_PREFIX/$action"
    TRAIN_INFER_DIR="${TRAIN_DIR}/train_infer"
    
    mkdir -p "$TRAIN_INFER_DIR"
    
    echo "Preparing train_infer for $action..."
    
    # Copy files to train_infer directory
    cp "${TRAIN_DIR}/train_facts.txt" "${TRAIN_INFER_DIR}/train_infer_facts.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_pos.txt" "${TRAIN_INFER_DIR}/train_infer_pos.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_neg.txt" "${TRAIN_INFER_DIR}/train_infer_neg.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_bk.txt" "${TRAIN_INFER_DIR}/train_infer_bk.txt" 2>/dev/null || true
    
    # Create query file (union of pos and neg examples)
    {
        cat "${TRAIN_DIR}/train_pos.txt" 2>/dev/null || true
        cat "${TRAIN_DIR}/train_neg.txt" 2>/dev/null || true
    } > "${TRAIN_INFER_DIR}/query_action.db"
    
    LOG_FILE="${MODEL_DIR}/action_train_infer.log"
    > "$LOG_FILE"
    
    echo "Running inference on training data for $action..."
    
    {
        echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Train Inference)"
        
        java -jar "$JAR" \
            -i \
            -model "$MODEL_DIR" \
            -test "$TRAIN_INFER_DIR" \
            -target "action" \
            -trees "$NUM_TREES" \
            -testNegPosRatio "$NEG_POS_RATIO" \
            -aucJarPath "$AUC_JAR"
        
        echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Train Inference)"
    } >> "$LOG_FILE" 2>&1
    
    echo "✅ Completed train inference for $action"
    echo ""
done

# ============================================================================
# STEP 4: CALIBRATION & EVALUATION
# ============================================================================
echo ""
echo "==========================================="
echo "STEP 4: Calibration and Evaluation"
echo "==========================================="
echo ""

python src/calibrate_and_evaluate.py \
    --model_base "$MODEL_BASE_PREFIX" \
    --data_base "$DATA_BASE" \
    --actions "${ACTIONS[@]}" \
    --seeds "${SEEDS[@]}"

echo ""
echo "=================================================="
echo "PIPELINE COMPLETE!"
echo "=================================================="
echo "Model directory: $MODEL_BASE_PREFIX"
echo "Results saved in respective action directories"
echo "=================================================="
