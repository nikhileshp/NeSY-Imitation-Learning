import subprocess
import json
import os
import argparse
import time

# Configuration
ACTIONS = ['noop', 'fire', 'up', 'right', 'left', 'down']
RATIO = 2.0
SEED = 42
MODEL_TYPE = 'resnet18' # or 'cnn'

def run_training(action):
    print(f"Running training for action: {action}")
    cmd = [
        "python", "train_per_action.py",
        "--action", action,
        "--seed", str(SEED),
        "--ratio", str(RATIO),
        "--model_type", MODEL_TYPE
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training {action}: {e}")
        return False

def aggregate_results():
    print("Aggregating results...")
    aggregated_data = {
        "action": [],
        "rgb_f1": [],
        "rgb_auc": [],
        "gaze_f1": [],
        "gaze_auc": []
    }
    
    action_map = {
        "noop": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5
    }
    
    for action in ACTIONS:
        # Path to results.json for this action
        # trained_models/seaquest/all/negpos_{ratio}_{model_str}/{action}/seed_{seed}/results.json
        model_name_str = f"rgb_{MODEL_TYPE}_64_32_bc"
        if MODEL_TYPE == 'cnn':
            model_name_str = "rgb_cnn_3_layers_64_32_bc"
            
        result_file = f"trained_models/seaquest/all/negpos_{int(RATIO)}_{model_name_str}/{action}/seed_{SEED}/results.json"
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            aggregated_data["action"].append(action_map[action])
            aggregated_data["rgb_f1"].append(data["rgb_f1"])
            aggregated_data["rgb_auc"].append(data["rgb_auc"])
            aggregated_data["gaze_f1"].append(data["gaze_f1"])
            aggregated_data["gaze_auc"].append(data["gaze_auc"])
        else:
            print(f"Warning: Results not found for {action} at {result_file}")
            
    # Save aggregated results
    output_file = f"results_ratio_{RATIO}.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated_data, f, indent=4)
        
    print(f"Aggregated results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_training", action="store_true", help="Skip training and just aggregate")
    args = parser.parse_args()
    
    if not args.skip_training:
        for action in ACTIONS:
            success = run_training(action)
            if not success:
                print(f"Stopping due to error in {action}")
                return
            
    aggregate_results()

if __name__ == "__main__":
    main()
