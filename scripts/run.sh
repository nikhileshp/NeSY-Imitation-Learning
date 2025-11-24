#!/bin/bash
set -e  # stop if any command fails

MAX_DEPTH=$1
NUM_TREES=$2
WEIGHTED=$3  # "true" or "false"
ONLY_TEST=$4  # "true" or "false"

if [ "$ONLY_TEST" == "true" ]; then
    echo "Skipping preprocess as ONLY_TEST is set to true."
    
else
    echo ""
    echo ""
    echo "Starting preprocess..."
    echo "Max tree depth: $MAX_DEPTH"
    echo "Weighted: $WEIGHTED"  
    echo "Abstraction: True"
    echo "ONLY_TEST: $ONLY_TEST"
    # python data/seaquest/preprocess.py --remove_0_weights --file relationships.txt --node_size 2 --max_tree_depth $MAX_DEPTH
# Common parameters
    echo "Preprocess completed."
fi

JAR="rdnboost/target/boostsrl-weights-2.0.0.jar"

AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils/"
TREES=$NUM_TREES
NEG_POS_RATIO=2
# List of targets (actions)
TARGETS=("fire" "up" "down" "left" "right" "noop")

TRAIN_DIRS=("data/seaquest/all/fire/train" "data/seaquest/all/up/train" "data/seaquest/all/down/train" "data/seaquest/all/left/train" "data/seaquest/all/right/train" "data/seaquest/all/noop/train")
# Corresponding model output directories (match 1-to-1 with TARGETS)
if $WEIGHTED == "true"; then
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
 
WEIGHTS=(
    "data/seaquest/all/fire/train/fact_weights.tsv"
    "data/seaquest/all/up/train/fact_weights.tsv"
    "data/seaquest/all/down/train/fact_weights.tsv"
    "data/seaquest/all/left/train/fact_weights.tsv"
    "data/seaquest/all/right/train/fact_weights.tsv"
    "data/seaquest/all/noop/train/fact_weights.tsv"
)
# Loop through each target/model 
if [ $WEIGHTED == "true" ] && [ $ONLY_TEST == "false" ]; then

    for i in "${!TARGETS[@]}"; do
        TARGET="${TARGETS[$i]}"
        MODEL="${MODELS[$i]}"
        TRAIN_DIR="${TRAIN_DIRS[$i]}"
        echo "======================================"
        echo "Running BoostSRL for target: $TARGET"
        echo "Saving model to: $MODEL"
        echo "======================================"
        java -jar "$JAR" \
            -l \
            -train "$TRAIN_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR" \
            -negPosRatio "$NEG_POS_RATIO" \
            -model "$MODEL" \

        echo "✅ Completed training for $TARGET"
        echo ""
    done
# If only testing is false and not weighted
elif [ $ONLY_TEST == "false" ]; then

    for i in "${!TARGETS[@]}"; do
        TARGET="${TARGETS[$i]}"
        MODEL="${MODELS[$i]}"
        TRAIN_DIR="${TRAIN_DIRS[$i]}"
        echo "======================================"
        echo "Running BoostSRL for target: $TARGET"
        echo "Saving model to: $MODEL"
        echo "======================================"
        java -jar "$JAR" \
            -l \
            -train "$TRAIN_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR" \
            -negPosRatio "$NEG_POS_RATIO" \
            -model "$MODEL" \

        echo "✅ Completed training for $TARGET"
        echo ""
    done

fi
echo "All runs finished successfully!"


TEST_DIRS=("data/seaquest/all/fire/test" "data/seaquest/all/up/test" "data/seaquest/all/down/test" "data/seaquest/all/left/test" "data/seaquest/all/right/test" "data/seaquest/all/noop/test")
 
 
# Corresponding log files

LOG_FILES=(

    "${MODEL_BASE}/fire/fire_infer.log"

    "${MODEL_BASE}/up/up_infer.log"

    "${MODEL_BASE}/down/down_infer.log"

    "${MODEL_BASE}/left/left_infer.log"

    "${MODEL_BASE}/right/right_infer.log"

    "${MODEL_BASE}/noop/noop_infer.log"

)
 
echo "🚀 Starting inference for all targets..."

echo ""
 
# Loop through targets

for i in "${!TARGETS[@]}"; do

    TARGET="${TARGETS[$i]}"

    MODEL="${MODELS[$i]}"

    TEST_DIR="${TEST_DIRS[$i]}"

    LOG_FILE="${LOG_FILES[$i]}"


 
    # Clear previous log if exists
> "$LOG_FILE"
 
    echo "======================================"

    echo "Running inference for target: $TARGET"

    echo "Model: $MODEL"

    echo "Test data: $TEST_DIR"

    echo "Log file: $LOG_FILE"

    echo "======================================"
 
    # Run inference and save to the target-specific log

    {

        echo "[START] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET"

        java -jar "$JAR" \
            -i \
            -model "$MODEL" \
            -test "$TEST_DIR" \
            -target "$TARGET" \
            -trees "$TREES" \
            -aucJarPath "$AUC_JAR" \
            

        echo "[END] $(date '+%Y-%m-%d %H:%M:%S') - Target: $TARGET"

        echo ""

    } >> "$LOG_FILE" 2>&1
 
    # Print the last 10 lines of this target’s log to console

    echo "🔍 Last 10 lines for target '$TARGET':"

    tail -n 10 "$LOG_FILE"

    echo ""

done

python experiments/eval.py --model_dir "$MODEL_BASE"