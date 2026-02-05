
import os
import re
import csv
import json

BASE_DIRS = [
    "trained_models/seaquest/all",
    "rdn_models/seaquest/all_pi",
    "rdn_models/seaquest/all"
]
OUTPUT_CSV = "experiment_results.csv"

def parse_folder_name(folder_name):
    """
    Parses parameters from the folder name.
    """
    params = {}
    params["experiment"] = folder_name
    
    # Defaults
    params["num_trees"] = "N/A"
    params["depth"] = "N/A"
    params["train_neg_pos"] = "N/A"
    params["test_neg_pos"] = "N/A" # Usually same as train or default
    params["model_type"] = "Unknown"
    params["grounding_penalty"] = "N/A"
    params["lambda"] = "N/A"
    
    parts = folder_name.split("_")
    
    # Lambda
    if "lambda" in parts:
        try:
            idx = parts.index("lambda")
            params["lambda"] = float(parts[idx+1])
            params["model_type"] = "RRT (PI)"
        except:
            pass
    
    # Train Neg/Pos
    if "negpos" in parts:
        try:
            idx = parts.index("negpos")
            params["train_neg_pos"] = float(parts[idx+1])
        except:
            pass
            
    # Trees
    if "trees" in parts:
        try:
            idx = parts.index("trees")
            params["num_trees"] = int(parts[idx+1])
        except:
            pass
            
    # Depth
    if "depth" in parts:
        try:
            idx = parts.index("depth")
            params["depth"] = int(parts[idx+1])
        except:
            pass
            
    # Grounding Penalty
    if "grounding" in parts and "penalty" in parts:
         try:
            idx = parts.index("penalty")
            params["grounding_penalty"] = float(parts[idx+1])
         except:
            pass

    # Model Type Detection (Priority to PI if already set)
    if params["model_type"] == "RRT (PI)":
        pass
    elif "rgb" in parts and "resnet18" in parts:
        params["model_type"] = "CNN ResNet"
    elif "rgb" in parts and "cnn" in parts:
        params["model_type"] = "CNN"
    elif "mlp" in parts:
        params["model_type"] = "MLP"
    elif "trees" in parts:
        num_trees = params["num_trees"]
        penalty = params["grounding_penalty"]
        
        if num_trees == 1:
            if penalty != "N/A":
                params["model_type"] = "RRT_w_regularization"
            else:
                params["model_type"] = "RRT"
        elif isinstance(num_trees, int) and num_trees > 1:
            if penalty != "N/A":
                params["model_type"] = "BRRT_w_regularization"
            else:
                params["model_type"] = "BRRT"
    
    return params

def get_metrics_from_log(log_path):
    metrics = {"f1": "N/A", "auc_pr": "N/A", "auc_roc": "N/A"}
    if not os.path.exists(log_path):
        return metrics
        
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
            # F1
            # Try standard format
            m = re.search(r"F1:\s+([0-9.]+)", content)
            if m: metrics["f1"] = float(m.group(1))
            else:
                # Try WILL format
                m = re.search(r"%\s+F1\s+=\s+([0-9.]+)", content)
                if m: metrics["f1"] = float(m.group(1))

            # AUC PR
            m = re.search(r"AUC PR:\s+([0-9.]+)", content)
            if m: metrics["auc_pr"] = float(m.group(1))
            else:
                m = re.search(r"%\s+AUC PR\s+=\s+([0-9.]+)", content)
                if m: metrics["auc_pr"] = float(m.group(1))

            # AUC ROC
            m = re.search(r"AUC ROC:\s+([0-9.]+)", content)
            if m: metrics["auc_roc"] = float(m.group(1))
            else:
                m = re.search(r"%\s+AUC ROC\s+=\s+([0-9.]+)", content)
                if m: metrics["auc_roc"] = float(m.group(1))
                
    except Exception as e:
        print(f"Error reading log {log_path}: {e}")
        
    return metrics

def get_metrics_from_json(json_path):
    metrics = {"f1": "N/A", "auc_pr": "N/A", "auc_roc": "N/A"}
    if not os.path.exists(json_path):
        return metrics
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Keys might vary, standardizing based on compare_all_models.py
            # But compare_all_models.py mostly looked at rgb_f1 etc.
            # Let's look for standard keys first
            
            metrics["f1"] = data.get("f1", data.get("rgb_f1", "N/A"))
            metrics["auc_pr"] = data.get("auc_pr", data.get("rgb_auc", "N/A")) # JSON in train_per_action uses 'rgb_auc' for PR
            metrics["auc_roc"] = data.get("auc_roc", "N/A")
            
    except Exception as e:
        print(f"Error reading json {json_path}: {e}")
        
    return metrics


def main():
    csv_rows = []
    
    for base_dir in BASE_DIRS:
        if not os.path.exists(base_dir):
            print(f"Base Directory {base_dir} does not exist! Skipping.")
            continue

        # Get all subdirectories
        for folder_name in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # -- Filter Logic -- 
            # Removed per user request to aggregate all runs
            # has_penalty = "grounding_penalty" in folder_name
            # is_new = "_new" in folder_name
            
            # if has_penalty and not is_new:
            #    continue
                
            print(f"Processing folder: {folder_name}")
            
            # Parse common parameters
            params = parse_folder_name(folder_name)
            
            # Iterate over ACTIONS (sub-folders)
            # We need to find where the results are. Usually .../{action}/seed_{seed}/...
            # List actions
            for action in os.listdir(folder_path):
                action_path = os.path.join(folder_path, action)
                if not os.path.isdir(action_path):
                    continue
                    
                # Iterate over SEEDS
                for seed_folder in os.listdir(action_path): # e.g. seed_42
                    if not seed_folder.startswith("seed_"):
                        continue
                    
                    seed = seed_folder.replace("seed_", "")
                    seed_path = os.path.join(action_path, seed_folder)
                    
                    # Extract Metrics
                    metrics = {"f1": "N/A", "auc_pr": "N/A", "auc_roc": "N/A"}
                    
                    # Try JSON (RGB/MLP often use JSON)
                    json_path = os.path.join(seed_path, "results.json")
                    
                    if os.path.exists(json_path):
                        json_metrics = get_metrics_from_json(json_path)
                        metrics.update(json_metrics)
                    
                    # If JSON missing or incomplete (N/A), try LOG
                    # RDN/BoostSRL usually uses test_infer.log
                    if metrics["f1"] == "N/A" or metrics["auc_pr"] == "N/A":
                         log_path = os.path.join(seed_path, f"test_infer_seed_{seed}.log")
                         if not os.path.exists(log_path):
                             log_path = os.path.join(seed_path, "test_infer.log")
                         
                         log_metrics = get_metrics_from_log(log_path)
                         
                         # Merge (prefer existing valid values from JSON if any)
                         for k, v in log_metrics.items():
                             if metrics.get(k, "N/A") == "N/A":
                                 metrics[k] = v
                                 
                    # Build Row
                    row = {
                        "Experiment": folder_name,
                        "Action": action,
                        "Seed": seed,
                        "Parameters (Num Trees)": params["num_trees"],
                        "(Depth)": params["depth"],
                        "Train Neg-to-Pos": params["train_neg_pos"],
                        "Test Neg-to-Pos": params["train_neg_pos"], # Assuming match
                        "Model": params["model_type"],
                        "Lambda": params["lambda"],
                        "AUC-PR": metrics["auc_pr"],
                        "AUC-ROC": metrics["auc_roc"],
                        "F1": metrics["f1"],
                    }
                    csv_rows.append(row)

    # Write CSV
    headers = ["Experiment", "Action", "Seed", "Parameters (Num Trees)", "(Depth)", "Train Neg-to-Pos", "Test Neg-to-Pos", "Model", "Lambda", "AUC-PR", "AUC-ROC", "F1"]
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"CSV generation complete. Saved to {OUTPUT_CSV}")
    print(f"Total rows: {len(csv_rows)}")

if __name__ == "__main__":
    main()
