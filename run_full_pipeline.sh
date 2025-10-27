#!/bin/bash
set -e  # stop if any command fails

MAX_DEPTH=$1
NUM_TREES=$2
WEIGHTED=$3  # "true" or "false"
ONLY_TEST=$4  # "true" or "false"

if [ "$ONLY_TEST" == "true" ]; then
    echo "Skipping training as ONLY_TEST is set to true."
else
    echo ""
    echo ""
    echo "Starting pipeline..."
    echo "Max tree depth: $MAX_DEPTH"
    echo "Weighted: $WEIGHTED"  
    echo "ONLY_TEST: $ONLY_TEST"
fi

JAR="rdnboost/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils/"
TREES=$NUM_TREES
NEG_POS_RATIO=2

# List of targets (actions)
TARGETS=("fire" "up" "down" "left" "right" "noop")

TRAIN_DIRS=("data/seaquest/all/fire/train" "data/seaquest/all/up/train" "data/seaquest/all/down/train" "data/seaquest/all/left/train" "data/seaquest/all/right/train" "data/seaquest/all/noop/train")

# Set model base directory
if [ "$WEIGHTED" == "true" ]; then
    MODEL_BASE="rdn_models/seaquest/negpos_${NEG_POS_RATIO}_trees_${TREES}_depth_${MAX_DEPTH}_per_example_weight"
else
    MODEL_BASE="rdn_models/seaquest/negpos_${NEG_POS_RATIO}_trees_${TREES}_depth_${MAX_DEPTH}"
fi

MODELS=(
    "$MODEL_BASE/fire"
    "$MODEL_BASE/up"
    "$MODEL_BASE/down"
    "$MODEL_BASE/left"
    "$MODEL_BASE/right"
    "$MODEL_BASE/noop"
)

WEIGHTS=(
    "data/seaquest/all/fire/train/fact_weights.tsv"
    "data/seaquest/all/up/train/fact_weights.tsv"
    "data/seaquest/all/down/train/fact_weights.tsv"
    "data/seaquest/all/left/train/fact_weights.tsv"
    "data/seaquest/all/right/train/fact_weights.tsv"
    "data/seaquest/all/noop/train/fact_weights.tsv"
)

# ============================================================================
# STEP 1: TRAINING
# ============================================================================
if [ "$WEIGHTED" == "true" ] && [ "$ONLY_TEST" == "false" ]; then
    echo ""
    echo "=========================================="
    echo "STEP 1: Training models (WEIGHTED)"
    echo "=========================================="
    echo "Setting up weight files..."
    
    # Ensure weight files are in place for weighted training
    for i in "${!TARGETS[@]}"; do
        TRAIN_DIR="${TRAIN_DIRS[$i]}"
        
        # If backup files exist (from previous unweighted run), restore them
        if [ -f "${TRAIN_DIR}/train_pos_weights.txt.bak" ]; then
            mv "${TRAIN_DIR}/train_pos_weights.txt.bak" "${TRAIN_DIR}/train_pos_weights.txt"
        fi
        if [ -f "${TRAIN_DIR}/train_neg_weights.txt.bak" ]; then
            mv "${TRAIN_DIR}/train_neg_weights.txt.bak" "${TRAIN_DIR}/train_neg_weights.txt"
        fi
    done
    
    for i in "${!TARGETS[@]}"; do
        TARGET="${TARGETS[$i]}"
        MODEL="${MODELS[$i]}"
        TRAIN_DIR="${TRAIN_DIRS[$i]}"
        echo ""
        echo "Training $TARGET..."
        java -jar "$JAR" \
            -l \
            -train "$TRAIN_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR" \
            -negPosRatio "$NEG_POS_RATIO" \
            -model "$MODEL"
        echo "✅ Completed training for $TARGET"
    done

elif [ "$ONLY_TEST" == "false" ]; then
    echo ""
    echo "=========================================="
    echo "STEP 1: Training models (UNWEIGHTED)"
    echo "=========================================="
    echo "Removing weight files for unweighted training..."
    
    # Move weight files out of the way for unweighted training
    for i in "${!TARGETS[@]}"; do
        TRAIN_DIR="${TRAIN_DIRS[$i]}"
        
        # Backup weight files if they exist
        if [ -f "${TRAIN_DIR}/train_pos_weights.txt" ]; then
            mv "${TRAIN_DIR}/train_pos_weights.txt" "${TRAIN_DIR}/train_pos_weights.txt.bak"
        fi
        if [ -f "${TRAIN_DIR}/train_neg_weights.txt" ]; then
            mv "${TRAIN_DIR}/train_neg_weights.txt" "${TRAIN_DIR}/train_neg_weights.txt.bak"
        fi
    done
    
    for i in "${!TARGETS[@]}"; do
        TARGET="${TARGETS[$i]}"
        MODEL="${MODELS[$i]}"
        TRAIN_DIR="${TRAIN_DIRS[$i]}"
        echo ""
        echo "Training $TARGET..."
        java -jar "$JAR" \
            -l \
            -train "$TRAIN_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR" \
            -negPosRatio "$NEG_POS_RATIO" \
            -model "$MODEL"
        echo "✅ Completed training for $TARGET"
    done
fi

# ============================================================================
# STEP 2: TESTING
# ============================================================================
TEST_DIRS=("data/seaquest/all/fire/test" "data/seaquest/all/up/test" "data/seaquest/all/down/test" "data/seaquest/all/left/test" "data/seaquest/all/right/test" "data/seaquest/all/noop/test")

echo ""
echo "=========================================="
echo "STEP 2: Running inference on test data"
echo "=========================================="
echo ""

for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    MODEL="${MODELS[$i]}"
    TEST_DIR="${TEST_DIRS[$i]}"
    LOG_FILE="${MODEL}/${TARGET}_test_infer.log"
    
    # Create model directory if it doesn't exist
    mkdir -p "$MODEL"
    
    # Clear previous log if exists
    > "$LOG_FILE"
    
    echo "Testing $TARGET..."
    
    # Run inference and save to the target-specific log
    {
        echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET (Test)"
        java -jar "$JAR" \
            -i \
            -model "$MODEL" \
            -test "$TEST_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR"
            
        echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET (Test)"
    } >> "$LOG_FILE" 2>&1
    
    echo "✅ Completed test inference for $TARGET"
done

# ============================================================================
# STEP 3: TRAINING INFERENCE (for calibration)
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 3: Running inference on training data (for calibration)"
echo "=========================================="
echo ""

# Create train_infer directories and query files
for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    TRAIN_DIR="${TRAIN_DIRS[$i]}"
    TRAIN_INFER_DIR="${TRAIN_DIR}/train_infer"
    
    # Create directory
    mkdir -p "$TRAIN_INFER_DIR"
    
    # Copy necessary files
    echo "Preparing train_infer for $TARGET..."
    cp "${TRAIN_DIR}/train_facts.txt" "${TRAIN_INFER_DIR}/train_infer_facts.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_pos.txt" "${TRAIN_INFER_DIR}/train_infer_pos.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_neg.txt" "${TRAIN_INFER_DIR}/train_infer_neg.txt" 2>/dev/null || true
    cp "${TRAIN_DIR}/train_bk.txt" "${TRAIN_INFER_DIR}/train_infer_bk.txt" 2>/dev/null || true
    
    # Create query file (union of pos and neg examples)
    {
        if [ -f "${TRAIN_DIR}/train_pos.txt" ]; then
            cat "${TRAIN_DIR}/train_pos.txt"
        fi
        if [ -f "${TRAIN_DIR}/train_neg.txt" ]; then
            sed "s/^${TARGET}/!${TARGET}/" "${TRAIN_DIR}/train_neg.txt"
        fi
    } > "${TRAIN_INFER_DIR}/query_${TARGET}.db"
    
    echo "✅ Created query file with $(wc -l < ${TRAIN_INFER_DIR}/query_${TARGET}.db) queries"
done

echo ""
echo "Running training inference..."
echo ""

# Run inference for each target on training data
for i in "${!TARGETS[@]}"; do
    TARGET="${TARGETS[$i]}"
    MODEL="${MODELS[$i]}"
    TRAIN_INFER_DIR="${TRAIN_DIRS[$i]}/train_infer"
    LOG_FILE="${MODEL}/${TARGET}_train_infer.log"
    
    # Create model directory if it doesn't exist
    mkdir -p "$MODEL"
    
    # Clear previous log if exists
    > "$LOG_FILE"
    
    echo "Training inference for $TARGET..."
    
    # Run inference
    {
        echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET (Train Inference)"
        java -jar "$JAR" \
            -i \
            -model "$MODEL" \
            -test "$TRAIN_INFER_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR"
            
        echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET (Train Inference)"
    } >> "$LOG_FILE" 2>&1
    
    # Check if AUC file was created
    if [ -f "${TRAIN_INFER_DIR}/AUC/aucTemp.txt" ]; then
        NUM_PROBS=$(wc -l < "${TRAIN_INFER_DIR}/AUC/aucTemp.txt")
        echo "✅ Generated $NUM_PROBS probabilities for $TARGET"
    else
        echo "⚠️  Warning: No AUC file generated for $TARGET"
    fi
done

# ============================================================================
# STEP 4: EVALUATION (with calibration)
# ============================================================================
echo ""
echo "=========================================="
echo "STEP 4: Evaluating models"
echo "=========================================="
echo ""

python eval_calibrated.py --model_dir "$MODEL_BASE"

echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Model directory: $MODEL_BASE"
echo "  - Test inference logs: $MODEL_BASE/{action}/{action}_test_infer.log"
echo "  - Train inference logs: $MODEL_BASE/{action}/{action}_train_infer.log"
echo "  - Evaluation report: $MODEL_BASE/eval_report.txt"
echo ""
