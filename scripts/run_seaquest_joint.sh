#!/bin/bash
set -e  # stop if any command fails

# Check for correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "nesy-il" ]; then
    echo "Error: Conda environment 'nesy-il' is not active."
    echo "Current environment: $CONDA_DEFAULT_ENV"
    echo "Please run: conda activate nesy-il"
    exit 1
fi

# Parse command line arguments
MAX_DEPTH=$1
NUM_TREES=$2
DEBUG_MODE=$3  # "true" or "false" (optional, default: false)
TEST_ONLY=$4   # "true" or "false" (optional, default: false)
USE_SAMPLING=$5 # "true" or "false" (optional, default: false)
LAMBDA=$6      # Lambda value for PI (optional, default: 1.0)

if [ -z "$MAX_DEPTH" ] || [ -z "$NUM_TREES" ]; then
    echo "Usage: $0 <max_depth> <num_trees> [debug_mode] [test_only] [use_sampling] [lambda]"
    echo "Example: $0 3 10 true false false 1.0"
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

# Default use_sampling to false if not specified
if [ -z "$USE_SAMPLING" ]; then
    USE_SAMPLING="false"
fi

# Default lambda to 1.0 if not specified
if [ -z "$LAMBDA" ]; then
    LAMBDA=1.0
fi

echo "=================================================="
echo "Joint Training pipeline for Seaquest with PI"
echo "=================================================="
echo "Max tree depth: $MAX_DEPTH"
echo "Number of trees: $NUM_TREES"
echo "Debug mode: $DEBUG_MODE"
echo "Test only: $TEST_ONLY"
echo "Lambda: $LAMBDA"
echo ""

# Configuration
# Point to the joint training jar
JAR="rdnboost-joint/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost-joint/src/edu/wisc/cs/will/DataSetUtils/"
NEG_POS_RATIO=2
DATA_BASE="data/seaquest/all"

# Seeds to test with
SEEDS=(42) # Start with one seed for now

# Base model directory
MODEL_BASE_PREFIX="rdn_models/seaquest/joint_pi/negpos_${NEG_POS_RATIO}_trees_${NUM_TREES}_depth_${MAX_DEPTH}_lambda_${LAMBDA}"

if [ "$DEBUG_MODE" == "true" ]; then
    # DEBUG_FLAG="-debugScoring" 
    echo "Debug mode enabled (no specific flag set, relying on code prints)"
    DEBUG_FLAG="" 
else
    DEBUG_FLAG=""
fi

# Actions to train/test
ACTIONS=("fire" "up" "down" "left" "right" "noop")

echo "Testing model: $MODEL_BASE_PREFIX"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "######################################################################"
    echo "PROCESSING SEED: $SEED"
    echo "######################################################################"

    # STEP 1: TRAINING
    if [ "$TEST_ONLY" == "true" ]; then
        echo "Skipping Training"
    else
        echo "--- Step 1: Training ---"
        for action in "${ACTIONS[@]}"; do
            TRAIN_DIR="$DATA_BASE/$action/train"
            MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
            
            echo "Training $action (Seed: $SEED)..."
            mkdir -p "$MODEL_DIR"
            
            # Check if PI facts exist
            if [ ! -f "$TRAIN_DIR/train_facts_pi.txt" ]; then
                echo "Warning: PI facts not found for $action at $TRAIN_DIR/train_facts_pi.txt. Skipping PI flag for this action (or fail?)"
            fi

            java -Xmx8G \
                 -jar "$JAR" \
                 -l \
                 -train "$TRAIN_DIR" \
                 -target "action" \
                 -trees "$NUM_TREES" \
                 -aucJarPath "$AUC_JAR" \
                 -negPosRatio "$NEG_POS_RATIO" \
                 -model "$MODEL_DIR" \
                 -pi \
                 -piLambda "$LAMBDA" \
                 $DEBUG_FLAG \
                 > "${MODEL_DIR}/train.log" 2>&1
            
            echo "✅ Completed training for $action. Logs in ${MODEL_DIR}/train.log"
        done
    fi

    # STEP 2: TESTING
    echo "--- Step 2: Testing ---"
    for action in "${ACTIONS[@]}"; do
        TEST_DIR="data/seaquest/all/$action/test"
        MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
        LOG_FILE="${MODEL_DIR}/test_infer_seed_${SEED}.log"
        
        if [ ! -d "$MODEL_DIR/bRDNs" ]; then
            echo "⚠️  Skipping $action - model not found"
            continue
        fi
        
        echo "Testing $action..."
        > "$LOG_FILE"
        
        java -Xmx8G \
             -jar "$JAR" \
             -i \
             -model "$MODEL_DIR" \
             -test "$TEST_DIR" \
             -target "action" \
             -trees "$NUM_TREES" \
             -testNegPosRatio "$NEG_POS_RATIO" \
             -aucJarPath "$AUC_JAR" \
             >> "$LOG_FILE" 2>&1
             
        echo "✅ Completed testing for $action"
    done
    
done
