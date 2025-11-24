#!/bin/bash

echo "Checking calibration pipeline progress..."
echo ""

ACTIONS=("fire" "up" "down" "left" "right" "noop")

for action in "${ACTIONS[@]}"; do
    TRAIN_INFER_DIR="data/seaquest/all/${action}/train/train_infer"
    AUC_FILE="${TRAIN_INFER_DIR}/AUC/aucTemp.txt"
    
    if [ -f "$AUC_FILE" ]; then
        NUM_LINES=$(wc -l < "$AUC_FILE")
        echo "✅ $action: $NUM_LINES predictions generated"
    else
        echo "⏳ $action: Still processing..."
    fi
done

echo ""
echo "Expected: 11937 predictions per action"
