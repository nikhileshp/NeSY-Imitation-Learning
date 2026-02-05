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
echo "Teacher Model Training (Privileged Data)"
echo "=================================================="
echo "Max tree depth: $MAX_DEPTH"
echo "Number of trees: $NUM_TREES"
echo "Grounding penalty: NONE"
echo ""

# Configuration
JAR="rdnboost/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils/"
NEG_POS_RATIO=2
DATA_BASE="data/seaquest/all_teacher"  # <--- Modified to point to teacher data

# Seeds to test with
SEEDS=(42)

# Base model directory
MODEL_BASE_PREFIX="rdn_models/seaquest/teacher_only/negpos_${NEG_POS_RATIO}_trees_${NUM_TREES}_depth_${MAX_DEPTH}"

# Debug flag for Java
if [ "$DEBUG_MODE" == "true" ]; then
    DEBUG_FLAG="-debugScoring"
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
echo ""

# ============================================================================
# MAIN LOOP
# ============================================================================

for SEED in "${SEEDS[@]}"; do
    echo "Processing SEED: $SEED"

    # STEP 1: TRAINING
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
            
            echo "✅ Completed training for $action"
        done
    fi
    echo ""

    # STEP 2: TESTING
    echo "--- Step 2: Testing ---"
    for action in "${ACTIONS[@]}"; do
        TEST_DIR="$DATA_BASE/$action/test"
        MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
        LOG_FILE="${MODEL_DIR}/test_infer_seed_${SEED}.log"
        
        if [ ! -d "$MODEL_DIR/bRDNs" ]; then
            echo "⚠️  Skipping $action - model not found"
            continue
        fi
        
        mkdir -p "$MODEL_DIR"
        > "$LOG_FILE"
        
        echo "Testing $action..."
        java -jar "$JAR" \
             -i \
             -model "$MODEL_DIR" \
             -test "$TEST_DIR" \
             -target "action" \
             -trees "$NUM_TREES" \
             -testNegPosRatio "$NEG_POS_RATIO" \
             -seed "$SEED" \
             -aucJarPath "$AUC_JAR" \
             > "$LOG_FILE" 2>&1
             
        echo "✅ Completed testing for $action"
    done
done

echo "Done."
