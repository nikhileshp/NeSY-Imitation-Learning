#!/bin/bash
set -e

# Config
JAR="rdnboost-pi/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost-pi/src/edu/wisc/cs/will/DataSetUtils/"
DATA_BASE="data/seaquest/all"
MODEL_BASE="rdn_models/seaquest/all_pi/negpos_2_trees_1_depth_3_lambda_0.5"
ACTIONS=("fire" "up" "down" "left" "right" "noop")
SEED=42
NUM_TREES=1

echo "Testing Teacher Models (PI_Model) for Seaquest"
echo "Model Base: $MODEL_BASE"

for action in "${ACTIONS[@]}"; do
    echo "Processing $action..."
    
    # Define paths
    MODEL_DIR="${MODEL_BASE}/${action}/seed_${SEED}/PI_Model"
    TEST_DIR="${DATA_BASE}/${action}/test"
    OUTPUT_LOG="${MODEL_BASE}/${action}/seed_${SEED}/test_infer_teacher_model_seed_${SEED}.log"
    
    # Check if model exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Error: Model directory not found: $MODEL_DIR"
        continue
    fi
    
    # Prepare model structure for inference (needs bRDNs/action.model)
    mkdir -p "${MODEL_DIR}/bRDNs"
    if [ -f "${MODEL_DIR}/model_PI.txt" ]; then
        cp "${MODEL_DIR}/model_PI.txt" "${MODEL_DIR}/bRDNs/action.model"
    else
         echo "Warning: model_PI.txt not found in $MODEL_DIR"
    fi
    
    echo "  Running inference..."
    java -Xmx4G -jar "$JAR" \
        -i \
        -model "$MODEL_DIR" \
        -test "$TEST_DIR" \
        -target "action" \
        -trees "$NUM_TREES" \
        -aucJarPath "$AUC_JAR" \
        > "$OUTPUT_LOG" 2>&1
        
    echo "  Saved log to: $OUTPUT_LOG"
done

echo "Done."
