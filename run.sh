#!/bin/bash
set -e  # stop if any command fails

python data/seaquest/preprocess.py --file data/seaquest/gaze_data_tmp/237_RZ_9656617_Feb-08-14-12-21_with_relationships_and_goals.txt
# Common parameters
JAR="rdnboost/target/boostsrl-weights-2.0.0.jar"

AUC_JAR="rdnboost/src/edu/wisc/cs/will/DataSetUtils/"
TREES=5
NEG_POS_RATIO=2
 
# List of targets (actions)
TARGETS=("fire" "up" "down" "left" "right" "noop")
TRAIN_DIRS=("data/seaquest/fire/train" "data/seaquest/up/train" "data/seaquest/down/train" "data/seaquest/left/train" "data/seaquest/right/train" "data/seaquest/noop/train")
# Corresponding model output directories (match 1-to-1 with TARGETS)
MODEL_BASE="rdn_models/seaquest/negpos_${NEG_POS_RATIO}_trees_${TREES}_all"
MODELS=(
    "$MODEL_BASE/fire"
    "$MODEL_BASE/up"
    "$MODEL_BASE/down"
    "$MODEL_BASE/left"
    "$MODEL_BASE/right"
    "$MODEL_BASE/noop"
)
 
WEIGHTS=(
    "data/seaquest/fire/fact_weights.tsv"
    "data/seaquest/up/fact_weights.tsv"
    "data/seaquest/down/fact_weights.tsv"
    "data/seaquest/left/fact_weights.tsv"
    "data/seaquest/right/fact_weights.tsv"
    "data/seaquest/noop/fact_weights.tsv"
)
# Loop through each target/model pair
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
        -factWeights "$WEIGHTS" \
        -model "$MODEL"
    echo "✅ Completed training for $TARGET"
    echo ""
done
 
echo "🎉 All runs finished successfully!"


TEST_DIRS=("data/seaquest/fire/test" "data/seaquest/up/test" "data/seaquest/down/test" "data/seaquest/left/test" "data/seaquest/right/test" "data/seaquest/noop/test")
 
 
# Corresponding log files

LOG_FILES=(

    "fire_infer.log"

    "up_infer.log"

    "down_infer.log"

    "left_infer.log"

    "right_infer.log"

    "noop_infer.log"

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
 
echo "🎉 All inference runs completed!"

echo "Logs stored in:"

for log in "${LOG_FILES[@]}"; do

    echo "  - $log"

done

python eval.py