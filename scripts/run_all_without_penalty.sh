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
    echo "Example: $0 3 10 true          # Enable debug mode"
    echo "Example: $0 3 10 false true    # Test only (skip training)"
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
echo "Training/Testing pipeline for 54_RZ_2461867"
echo "=================================================="
echo "Max tree depth: $MAX_DEPTH"
echo "Number of trees: $NUM_TREES"
echo "Debug mode: $DEBUG_MODE"
echo "Test only: $TEST_ONLY"
echo "Grounding penalty: NONE"
echo ""

# Configuration
JAR="rdnboost/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils/"
NEG_POS_RATIO=1
DATA_BASE="data/seaquest/all"

# Seeds to test with (can be modified to test different random samples)
SEEDS=(42 123 456 789 1729)


# Base model directory (will be extended with seed info)
MODEL_BASE_PREFIX="rdn_models/seaquest/all/negpos_${NEG_POS_RATIO}_trees_${NUM_TREES}_depth_${MAX_DEPTH}"

# Debug flag for Java
if [ "$DEBUG_MODE" == "true" ]; then
    DEBUG_FLAG="-debugScoring"
    echo "Debug scoring enabled - verbose output will be generated"
else
    DEBUG_FLAG=""
fi

# Actions to train/test
ACTIONS=("fire" "up" "down" "left" "right" "noop")

# Update background knowledge with max_depth (only if not test only)
if [ "$TEST_ONLY" != "true" ]; then
    echo "Updating background knowledge files with max_depth=$MAX_DEPTH..."
    python src/change_bk.py --max_depth "$MAX_DEPTH" --base_dir "$DATA_BASE"
fi

echo "Testing model: $MODEL_BASE_PREFIX"
echo "Testing configuration: $(basename $MODEL_BASE_PREFIX)"
echo ""

# ============================================================================
# MAIN LOOP: Iterate Seed by Seed
# ============================================================================

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "PROCESSING SEED: $SEED"
    echo "######################################################################"
    echo ""

    # ============================================================================
    # STEP 1: TRAINING
    # ============================================================================
    if [ "$TEST_ONLY" == "true" ]; then
        echo "Skipping Training (TEST_ONLY=true)"
    else
        echo "--- Step 1: Training ---"
        for action in "${ACTIONS[@]}"; do
            TRAIN_DIR="$DATA_BASE/$action/train"
            MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
            
            echo "Training $action (Seed: $SEED)..."
            mkdir -p "$MODEL_DIR"
            
            java -jar "$JAR" \
                 -l \
                 -train "$TRAIN_DIR" \
                 -target "action" \
                 -trees "$NUM_TREES" \
                 -aucJarPath "$AUC_JAR" \
                 -negPosRatio "$NEG_POS_RATIO" \
                 -model "$MODEL_DIR" \
                 -seed "$SEED" \
                 $DEBUG_FLAG
            
            echo "✅ Completed training for $action (Seed: $SEED)"
        done
    fi
    echo ""

    # # ============================================================================
    # # STEP 2: TESTING (Unified Test Set, All Examples)
    # # ============================================================================
    # echo "--- Step 2: Testing on Unified Test Set (All Examples) ---"
    # # Use -1 for all examples (disable neg/pos ratio subsampling)
    
    
    # for action in "${ACTIONS[@]}"; do
    #     # Use unified test set
    #     TEST_DIR="data/seaquest/all/$action/test"
    #     MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
    #     LOG_FILE="${MODEL_DIR}/test_infer_seed_${SEED}.log"
        
    #     # Check if model exists
    #     if [ ! -d "$MODEL_DIR/bRDNs" ]; then
    #         echo "⚠️  Skipping $action - model not found at $MODEL_DIR"
    #         continue
    #     fi
        
    #     mkdir -p "$MODEL_DIR"
    #     > "$LOG_FILE"
        
    #     echo "Testing $action (Seed: $SEED)..."
        
    #     {
    #         echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test, Seed: $SEED)"
    #         java -jar "$JAR" \
    #              -i \
    #              -model "$MODEL_DIR" \
    #              -test "$TEST_DIR" \
    #              -target "action" \
    #              -trees "$NUM_TREES" \
    #              -testNegPosRatio "$NEG_POS_RATIO" \
    #              -seed "$SEED" \
    #              -aucJarPath "$AUC_JAR"
    #         echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test, Seed: $SEED)"
    #     } >> "$LOG_FILE" 2>&1
        
    #     # Move AUC results to model directory to prevent overwriting
    #     if [ -d "$TEST_DIR/AUC" ]; then
    #         rm -rf "$MODEL_DIR/test_AUC"
    #         mv "$TEST_DIR/AUC" "$MODEL_DIR/test_AUC"
    #     fi
        
    #     echo "✅ Completed test inference for $action (Seed: $SEED)"
    # done
    # echo ""

    # ============================================================================
    # STEP 3: TRAINING INFERENCE (for calibration)
    # ============================================================================
    echo "--- Step 3: Training Inference (for Calibration) ---"
    
    for action in "${ACTIONS[@]}"; do
        TRAIN_DIR="$DATA_BASE/$action/train"
        MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
        TRAIN_INFER_DIR="${MODEL_DIR}/train_infer"
        
        # Check if model exists
        if [ ! -d "$MODEL_DIR/bRDNs" ]; then
             continue
        fi

        mkdir -p "$TRAIN_INFER_DIR"
        
        # Prepare data
        cp "${TRAIN_DIR}/train_facts.txt" "${TRAIN_INFER_DIR}/train_infer_facts.txt" 2>/dev/null || true
        cp "${TRAIN_DIR}/train_pos.txt" "${TRAIN_INFER_DIR}/train_infer_pos.txt" 2>/dev/null || true
        cp "${TRAIN_DIR}/train_neg.txt" "${TRAIN_INFER_DIR}/train_infer_neg.txt" 2>/dev/null || true
        cp "${TRAIN_DIR}/train_bk.txt" "${TRAIN_INFER_DIR}/train_infer_bk.txt" 2>/dev/null || true
        
        # Create query file (union of pos and neg examples)
        {
            cat "${TRAIN_DIR}/train_pos.txt" 2>/dev/null || true
            cat "${TRAIN_DIR}/train_neg.txt" 2>/dev/null || true
        } > "${TRAIN_INFER_DIR}/query_action.db"
        
        LOG_FILE="${MODEL_DIR}/train_infer.log"
        > "$LOG_FILE"
        
        echo "Running inference on training data for $action (Seed: $SEED)..."
        
        {
            echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Train Inference)"
            java -jar "$JAR" \
                 -i \
                 -model "$MODEL_DIR" \
                 -test "$TRAIN_INFER_DIR" \
                 -target "action" \
                 -trees "$NUM_TREES" \
                 -testNegPosRatio "$NEG_POS_RATIO" \
                 -seed "$SEED" \
                 -aucJarPath "$AUC_JAR"
            echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Train Inference)"
        } >> "$LOG_FILE" 2>&1
        
        echo "✅ Completed train inference for $action (Seed: $SEED)"
    done
    echo ""

    # ============================================================================
    # STEP 4: EVALUATION (Combined Test)
    # ============================================================================
    echo "--- Step 4: Evaluation (Combined Test) ---"
    

    TEST_NEG_POS_RATIO=-1
    
    for action in "${ACTIONS[@]}"; do
        # Use unified test set
        TEST_DIR="data/seaquest/all/$action/test"
        MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
        LOG_FILE="${MODEL_DIR}/test_infer_all_seed_${SEED}.log"
        
        # Check if model exists
        if [ ! -d "$MODEL_DIR/bRDNs" ]; then
            echo "⚠️  Skipping $action - model not found at $MODEL_DIR"
            continue
        fi
        
        mkdir -p "$MODEL_DIR"
        > "$LOG_FILE"
        
        echo "Testing $action (Seed: $SEED)..."
        
        {
            echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test, Seed: $SEED)"
            java -jar "$JAR" \
                 -i \
                 -model "$MODEL_DIR" \
                 -test "$TEST_DIR" \
                 -target "action" \
                 -trees "$NUM_TREES" \
                 -testNegPosRatio "$TEST_NEG_POS_RATIO" \
                 -seed "$SEED" \
                 -aucJarPath "$AUC_JAR"
            echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test, Seed: $SEED)"
        } >> "$LOG_FILE" 2>&1
        
        # Move AUC results to model directory to prevent overwriting
        if [ -d "$TEST_DIR/AUC" ]; then
            rm -rf "$MODEL_DIR/test_AUC"
            mv "$TEST_DIR/AUC" "$MODEL_DIR/test_AUC"
        fi
        
        echo "✅ Completed test inference for $action for every example (Seed: $SEED)"
    done

    echo ""
    python experiments/eval_calibrated.py \
        --model_dir "$MODEL_BASE_PREFIX" \
        --data_base "$DATA_BASE" \
        --seeds "$SEED" \
        --output_file "eval_report_seed_$SEED.txt"

    echo "✅ Evaluation completed for Seed: $SEED"
    echo "Results saved to: $MODEL_BASE_PREFIX/eval_report_seed_$SEED.txt"
    echo ""

done

echo "=================================================="
echo "Pipeline completed successfully!"
echo "=================================================="