import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# Configuration
TRAINED_MODELS_DIR = "trained_models/seaquest/all"
ACTIONS = ["noop", "fire", "up", "right", "left", "down"]
RATIO = 2.0
SEED = 42

# Model Definitions
MODELS = [
    {
        "name": "RGB (ResNet)",
        "type": "rgb",
        "path_pattern": f"negpos_{int(RATIO)}_rgb_resnet18_64_32_bc",
        "color": "red"
    },
    {
        "name": "RGB (CNN)",
        "type": "rgb",
        "path_pattern": f"negpos_{int(RATIO)}_rgb_cnn_3_layers_64_32_bc",
        "color": "orange"
    },
    {
        "name": "RGB (ResNet + Gaze)",
        "type": "rgb_gaze",
        "path_pattern": f"negpos_{int(RATIO)}_rgb_resnet18_64_32_bc",
        "color": "darkred"
    },
    {
        "name": "RGB (CNN + Gaze)",
        "type": "rgb_gaze",
        "path_pattern": f"negpos_{int(RATIO)}_rgb_cnn_3_layers_64_32_bc",
        "color": "darkorange"
    },
    {
        "name": "MLP (BC)",
        "type": "log",
        "path_pattern": f"negpos_{int(RATIO)}_mlp_64_32_bc",
        "color": "purple"
    },
    {
        "name": "RDN (Depth 1)",
        "type": "log",
        "path_pattern": f"negpos_{int(RATIO)}_trees_1_depth_1",
        "color": "blue"
    },
    {
        "name": "RDN (Depth 2)",
        "type": "log",
        "path_pattern": f"negpos_{int(RATIO)}_trees_1_depth_2",
        "color": "cyan"
    },
    {
        "name": "RDN (Depth 3)",
        "type": "log",
        "path_pattern": f"negpos_{int(RATIO)}_trees_1_depth_3",
        "color": "green"
    },
    # {
    #     "name": "RDN (Depth 3 + Gaze)",
    #     "type": "log",
    #     "path_pattern": f"negpos_{int(RATIO)}_trees_1_depth_3_grounding_penalty_0.1",
    #     "color": "magenta"
    # }
]

def get_rgb_metrics(model_dir, metric_prefix="rgb"):
    json_file = os.path.join(model_dir, "results.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            return {
                "f1": data.get(f"{metric_prefix}_f1", 0.0),
                "auc": data.get(f"{metric_prefix}_auc", 0.0)
            }
    return {"f1": 0.0, "auc": 0.0}

def get_log_metrics(model_dir):
    # Try standard log file first
    log_file = os.path.join(model_dir, "test_infer.log")
    
    # If not found, try seed-specific log file (common in gaze models)
    if not os.path.exists(log_file):
        log_file = os.path.join(model_dir, f"test_infer_seed_{SEED}.log")
        
    metrics = {"f1": 0.0, "auc": 0.0}
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Parse F1
            # Try standard format: "F1: 0.5233"
            match_f1 = re.search(r"F1:\s+([0-9.]+)", content)
            if not match_f1:
                # Try gaze/WILL format: "%   F1        = 0.532782"
                match_f1 = re.search(r"%\s+F1\s+=\s+([0-9.]+)", content)
                
            if match_f1:
                metrics["f1"] = float(match_f1.group(1))
            
            # Parse AUC PR
            # Try standard format: "AUC PR: 0.5772"
            match_auc = re.search(r"AUC PR:\s+([0-9.]+)", content)
            if not match_auc:
                # Try gaze/WILL format: "%   AUC PR    = 0.496365"
                match_auc = re.search(r"%\s+AUC PR\s+=\s+([0-9.]+)", content)
                
            if match_auc:
                metrics["auc"] = float(match_auc.group(1))
                
    return metrics

def collect_results():
    results = {model["name"]: {"f1": [], "auc": []} for model in MODELS}
    
    for action in ACTIONS:
        for model in MODELS:
            model_dir = os.path.join(TRAINED_MODELS_DIR, model["path_pattern"], action, f"seed_{SEED}")
            
            if model["type"] == "rgb":
                metrics = get_rgb_metrics(model_dir, "rgb")
            elif model["type"] == "rgb_gaze":
                metrics = get_rgb_metrics(model_dir, "gaze")
            else:
                metrics = get_log_metrics(model_dir)
                
            results[model["name"]]["f1"].append(metrics["f1"])
            results[model["name"]]["auc"].append(metrics["auc"])
            
    return results

def plot_results(results):
    x = np.arange(len(ACTIONS))
    width = 0.12 # Adjust width based on number of models
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # F1 Score
    ax = axes[0]
    for i, model in enumerate(MODELS):
        offset = (i - len(MODELS)/2) * width + width/2
        ax.bar(x + offset, results[model["name"]]["f1"], width, label=model["name"], color=model["color"])
        
    ax.set_ylabel('F1 Score')
    ax.set_title(f'F1 Score per Action (Ratio {RATIO})')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(MODELS))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # AUC-PR
    ax = axes[1]
    for i, model in enumerate(MODELS):
        offset = (i - len(MODELS)/2) * width + width/2
        ax.bar(x + offset, results[model["name"]]["auc"], width, label=model["name"], color=model["color"])
        
    ax.set_ylabel('AUC-PR')
    ax.set_title(f'AUC-PR per Action (Ratio {RATIO})')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(MODELS))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('comparison_all_models.png')
    print("Saved comparison plot to comparison_all_models.png")

def main():
    results = collect_results()
    plot_results(results)

if __name__ == "__main__":
    main()
