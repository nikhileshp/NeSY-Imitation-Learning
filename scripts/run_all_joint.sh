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
echo "Joint Training Pipeline (Teacher-Student) for 54_RZ_2461867"
echo "=================================================="
echo "Max tree depth: $MAX_DEPTH"
echo "Number of trees: $NUM_TREES"
echo "Debug mode: $DEBUG_MODE"
echo "Test only: $TEST_ONLY"
echo ""

# Configuration
# Point to the compiled jar
JAR="rdnboost-joint/target/boostsrl-weights-2.0.0.jar"
# AUC jar path
AUC_JAR="rdnboost-joint/src/edu/wisc/cs/will/DataSetUtils/"
NEG_POS_RATIO=2
DATA_BASE="data/seaquest/all"

# Teacher Configuration
TEACHER_BASE_DIR="rdn_models/seaquest/teacher_only/negpos_2_trees_1_depth_3"
LAMBDA=1.0

# Seeds to test with
SEEDS=(42) # Start with seed 42 as per requirement

# Base model directory for JOINT training
MODEL_BASE_PREFIX="rdn_models/seaquest/joint/negpos_${NEG_POS_RATIO}_trees_${NUM_TREES}_depth_${MAX_DEPTH}_lambda_${LAMBDA}"

# Debug flag for Java
if [ "$DEBUG_MODE" == "true" ]; then
    DEBUG_FLAG="-debugScoring"
    echo "Debug scoring enabled - verbose output will be generated"
else
    DEBUG_FLAG=""
fi

# Actions to train/test
ACTIONS=("fire")

echo "Training model: $MODEL_BASE_PREFIX"
echo "Using Teacher Models from: $TEACHER_BASE_DIR"
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
        echo "--- Step 1: Training (Joint with Teacher) ---"
        for action in "${ACTIONS[@]}"; do
            TRAIN_DIR="$DATA_BASE/$action/train"
            MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
            
            # Construct Teacher Directory for this action/seed
            # Teacher path: rdn_models/seaquest/teacher_only/negpos_2_trees_1_depth_3/fire/seed_42
            TEACHER_DIR="$TEACHER_BASE_DIR/$action/seed_$SEED"
            
            # Verify teacher exists
            if [ ! -d "$TEACHER_DIR" ]; then
                echo "⚠️  Teacher model not found at $TEACHER_DIR. Skipping $action."
                continue
            fi

            echo "Training $action (Seed: $SEED) with Teacher at $TEACHER_DIR..."
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
                 -teacherModel "$TEACHER_DIR" \
                 -pi \
                 -piLambda "$LAMBDA" \
                 $DEBUG_FLAG
            
            echo "✅ Completed joint training for $action (Seed: $SEED)"
        done
    fi
    echo ""

    # ============================================================================
    # STEP 2: TESTING
    # ============================================================================
    echo "--- Step 2: Testing ---"
    
    for action in "${ACTIONS[@]}"; do
        TEST_DIR="data/seaquest/all/$action/test"
        MODEL_DIR="$MODEL_BASE_PREFIX/$action/seed_$SEED"
        LOG_FILE="${MODEL_DIR}/test_infer_seed_${SEED}.log"
        
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
                 -testNegPosRatio "$NEG_POS_RATIO" \
                 -seed "$SEED" \
                 -aucJarPath "$AUC_JAR"
            echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test, Seed: $SEED)"
        } >> "$LOG_FILE" 2>&1
        
        echo "✅ Completed test inference for $action (Seed: $SEED)"
    done
    echo ""

done

echo "=================================================="
echo "Pipeline completed successfully!"
echo "=================================================="
