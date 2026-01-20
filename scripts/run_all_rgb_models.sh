#!/bin/bash

# Define actions and models
ACTIONS=("fire" "up" "down" "left" "right" "noop")
MODELS=("resnet18" "cnn")

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting training for all models and actions..."
echo "Timestamp: $(date)"

for model in "${MODELS[@]}"; do
    for action in "${ACTIONS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Training Model: $model, Action: $action"
        echo "----------------------------------------------------------------"
        
        # Run training
        if python train_per_action.py --action "$action" --model_type "$model" --epochs 50; then
            echo "Successfully trained $model for $action"
        else
            echo "Error training $model for $action"
        fi
        
        echo ""
    done
done

echo "All training runs completed at $(date)."
