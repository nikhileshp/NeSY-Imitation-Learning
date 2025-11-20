#!/bin/bash
set -e  # stop if any command fails

# Parse command line arguments
MAX_DEPTH=$1
NUM_TREES=$2
DEBUG_MODE=$3  # "true" or "false" (optional, default: false)

if [ -z "$MAX_DEPTH" ] || [ -z "$NUM_TREES" ]; then
    echo "Usage: $0 <max_depth> <num_trees> [debug_mode]"
    echo "Example: $0 3 10"
    echo "Example: $0 3 10 true    # Enable debug mode"
    exit 1
fi

# Default debug mode to false if not specified
if [ -z "$DEBUG_MODE" ]; then
    DEBUG_MODE="false"
fi

echo "=================================================="
echo "Training/Testing pipeline for 54_RZ_2461867"
echo "=================================================="
echo "Max tree depth: $MAX_DEPTH"
echo "Number of trees: $NUM_TREES"
echo "Debug mode: $DEBUG_MODE"
echo "Grounding penalty: threshold=0.7, alpha=0.1, beta=0.5, strategy=min"
echo ""

# Configuration
JAR="rdnboost/target/boostsrl-1.1.1.jar"
AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils/"
NEG_POS_RATIO=2
DATA_BASE="data/seaquest/single_t/54_RZ_2461867"
MODEL_BASE="rdn_models/seaquest/single_t/54_RZ_2461867/negpos_${NEG_POS_RATIO}_trees_${NUM_TREES}_depth_${MAX_DEPTH}_grounding_penalty"

# Grounding penalty parameters
GROUNDING_THRESHOLD=0.7
GROUNDING_ALPHA=0.1
GROUNDING_BETA=0.5
GROUNDING_STRATEGY="min"

# Debug flag for Java
if [ "$DEBUG_MODE" == "true" ]; then
    DEBUG_FLAG="-debugScoring"
    echo "Debug scoring enabled - verbose output will be generated"
else
    DEBUG_FLAG=""
fi

# Actions to train/test
ACTIONS=("fire" "up" "down" "left" "right" "noop")

# Update background knowledge with max_depth
echo "Updating background knowledge files with max_depth=$MAX_DEPTH..."
python change_bk.py --max_depth "$MAX_DEPTH" --base_dir "$DATA_BASE"

# ============================================================================
# STEP 1: TRAINING
# ============================================================================
echo ""
echo "==========================================="
echo "STEP 1: Training models with grounding penalty"
echo "==========================================="
echo ""

for action in "${ACTIONS[@]}"; do
    TRAIN_DIR="$DATA_BASE/$action/train"
    MODEL_DIR="$MODEL_BASE/$action"
    
    echo "Training $action..."
    mkdir -p "$MODEL_DIR"
    
    java -Dgrounding.penalty.threshold=$GROUNDING_THRESHOLD \
         -Dgrounding.penalty.alpha=$GROUNDING_ALPHA \
         -Dgrounding.penalty.beta=$GROUNDING_BETA \
         -Dgrounding.penalty.strategy=$GROUNDING_STRATEGY \
         -jar "$JAR" \
         -l \
         -train "$TRAIN_DIR" \
         -target "action" \
         -trees "$NUM_TREES" \
         -aucJarPath "$AUC_JAR" \
         -negPosRatio "$NEG_POS_RATIO" \
         -model "$MODEL_DIR" \
         -use-distance-weights \
         $DEBUG_FLAG
    
    echo "✅ Completed training for $action"
    echo ""
done

# ============================================================================
# STEP 2: TESTING
# ============================================================================
echo ""
echo "==========================================="
echo "STEP 2: Running inference on test data"
echo "==========================================="
echo ""

for action in "${ACTIONS[@]}"; do
    TEST_DIR="$DATA_BASE/$action/test"
    MODEL_DIR="$MODEL_BASE/$action"
    LOG_FILE="${MODEL_DIR}/action_test_infer.log"
    
    mkdir -p "$MODEL_DIR"
    > "$LOG_FILE"
    
    echo "Testing $action..."
    
    {
        echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test)"
        java -Dgrounding.penalty.threshold=$GROUNDING_THRESHOLD \
             -Dgrounding.penalty.alpha=$GROUNDING_ALPHA \
             -Dgrounding.penalty.beta=$GROUNDING_BETA \
             -Dgrounding.penalty.strategy=$GROUNDING_STRATEGY \
             -jar "$JAR" \
             -i \
             -model "$MODEL_DIR" \
             -test "$TEST_DIR" \
             -target "action" \
             -trees "$NUM_TREES" \
             -testNegPosRatio "$NEG_POS_RATIO" \
             -aucJarPath "$AUC_JAR"
        echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Action: $action (Test)"
    } >> "$LOG_FILE" 2>&1
    
    echo "✅ Completed test inference for $action"
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
    MODEL_DIR="$MODEL_BASE/$action"
    TRAIN_INFER_DIR="${TRAIN_DIR}/train_infer"
    
    mkdir -p "$TRAIN_INFER_DIR"
    
    echo "Preparing train_infer for $action..."
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
        java -Dgrounding.penalty.threshold=$GROUNDING_THRESHOLD \
             -Dgrounding.penalty.alpha=$GROUNDING_ALPHA \
             -Dgrounding.penalty.beta=$GROUNDING_BETA \
             -Dgrounding.penalty.strategy=$GROUNDING_STRATEGY \
             -jar "$JAR" \
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

python eval_calibrated.py \
    --model_dir "$MODEL_BASE" \
    --data_base "$DATA_BASE"

echo ""
echo "=================================================="
echo "Pipeline completed successfully!"
echo "=================================================="
echo "Results saved to: $MODEL_BASE"
echo "Evaluation report: $MODEL_BASE/eval_report.txt"
