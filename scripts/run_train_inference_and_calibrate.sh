#!/bin/bash
set -e  # stop if any command fails

MAX_DEPTH=$1
NUM_TREES=$2
WEIGHTED=$3  # "true" or "false"

echo "=========================================="
echo "Running Training Inference for Calibration"
echo "Max tree depth: $MAX_DEPTH"
echo "Number of trees: $NUM_TREES"
echo "Weighted: $WEIGHTED"
echo "=========================================="
echo ""

JAR="rdnboost/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils/"
TREES=$NUM_TREES
NEG_POS_RATIO=2

# List of targets (actions)
TARGETS=("fire" "up" "down" "left" "right" "noop")

TRAIN_DIRS=("data/seaquest/all/fire/train" "data/seaquest/all/up/train" "data/seaquest/all/down/train" "data/seaquest/all/left/train" "data/seaquest/all/right/train" "data/seaquest/all/noop/train")

# Set model base directory
if [ "$WEIGHTED" == "true" ]; then
    MODEL_BASE="rdn_models/seaquest/weighted_negpos_${NEG_POS_RATIO}_trees_${TREES}_depth_${MAX_DEPTH}_example_w"
else
    MODEL_BASE="rdn_models/seaquest/unweighted_negpos_${NEG_POS_RATIO}_trees_${TREES}_depth_${MAX_DEPTH}"
fi

MODELS=(
    "$MODEL_BASE/fire"
    "$MODEL_BASE/up"
    "$MODEL_BASE/down"
    "$MODEL_BASE/left"
    "$MODEL_BASE/right"
    "$MODEL_BASE/noop"
)

echo "🚀 Step 1: Running inference on training data for all targets..."
echo ""

# Create train_infer directories if they don't exist
for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    TRAIN_DIR="${TRAIN_DIRS[$i]}"
    TRAIN_INFER_DIR="${TRAIN_DIR}/train_infer"
    
    # Create directory
    mkdir -p "$TRAIN_INFER_DIR"
    
    # Copy necessary files from train to train_infer
    echo "Preparing train_infer directory for $TARGET..."
    cp "${TRAIN_DIR}/train_facts.txt" "${TRAIN_INFER_DIR}/train_infer_facts.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_pos.txt" "${TRAIN_INFER_DIR}/train_infer_pos.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_neg.txt" "${TRAIN_INFER_DIR}/train_infer_neg.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_bk.txt" "${TRAIN_INFER_DIR}/train_infer_bk.txt" 2>/dev/null || true
    
    # Create query file (union of pos and neg examples)
    echo "Creating query_${TARGET}.db for training data..."
    {
        if [ -f "${TRAIN_DIR}/train_pos.txt" ]; then
            cat "${TRAIN_DIR}/train_pos.txt"
        fi
        if [ -f "${TRAIN_DIR}/train_neg.txt" ]; then
            sed "s/^${TARGET}/!${TARGET}/" "${TRAIN_DIR}/train_neg.txt"
        fi
    } > "${TRAIN_INFER_DIR}/query_${TARGET}.db"
    
    echo "✅ Created query file with $(wc -l < ${TRAIN_INFER_DIR}/query_${TARGET}.db) queries"
    echo ""
done

echo "🔮 Step 2: Running inference on training data..."
echo ""

# Run inference for each target on training data
for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    MODEL="${MODELS[$i]}"
    TRAIN_INFER_DIR="${TRAIN_DIRS[$i]}/train_infer"
    LOG_FILE="${MODEL}/train_infer.log"
    
    # Create model directory if it doesn't exist
    mkdir -p "$MODEL"
    
    # Clear previous log if exists
    > "$LOG_FILE"
    
    echo "======================================"
    echo "Running inference for target: $TARGET"
    echo "Model: $MODEL"
    echo "Train infer data: $TRAIN_INFER_DIR"
    echo "Log file: $LOG_FILE"
    echo "======================================"
    
    # Run inference
    {
        echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET (Training Data)"
        java -jar "$JAR" \
            -i \
            -model "$MODEL" \
            -test "$TRAIN_INFER_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR"
            
        echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET (Training Data)"
        echo ""
    } >> "$LOG_FILE" 2>&1
    
    # Check if AUC file was created
    if [ -f "${TRAIN_INFER_DIR}/AUC/aucTemp.txt" ]; then
        NUM_PROBS=$(wc -l < "${TRAIN_INFER_DIR}/AUC/aucTemp.txt")
        echo "✅ Generated $NUM_PROBS probabilities for $TARGET"
    else
        echo "⚠️  Warning: No AUC file generated for $TARGET"
    fi
    
    echo ""
done

echo "🎯 Step 3: Training calibration model..."
echo ""

# Run the calibration script
python experiments/eval_calibrated.py --model_dir "$MODEL_BASE"

echo ""
echo "=========================================="
echo "✅ All steps completed successfully!"
echo "=========================================="
