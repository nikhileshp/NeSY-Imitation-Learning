import os
import re
import pandas as pd

def parse_log_file(filepath):
    """Extracts AUC ROC and AUC PR from a log file."""
    auc_roc = None
    auc_pr = None
    with open(filepath, 'r') as f:
        for line in f:
            if "AUC ROC" in line:
                match = re.search(r"AUC ROC\s*=\s*([0-9.]+)", line)
                if match:
                    auc_roc = float(match.group(1))
            if "AUC PR" in line:
                match = re.search(r"AUC PR\s*=\s*([0-9.]+)", line)
                if match:
                    auc_pr = float(match.group(1))
    return auc_roc, auc_pr

def aggregate_results():
    results = []
    
    # Paths to traverse
    # Format: (Model Name, Base Directory)
    sources = [
        ("Teacher Only", "rdn_models/seaquest/teacher_only/negpos_2_trees_1_depth_3"),
        ("Baseline (New)", "rdn_models/seaquest/all/negpos_2_trees_1_depth_3_new")
    ]
    
    for model_name, base_dir in sources:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} does not exist.")
            continue
            
        print(f"Scanning {base_dir} for {model_name}...")
        
        # Directory structure: base_dir / action / seed_X / test_infer_seed_X.log
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.startswith("test_infer_seed_") and file.endswith(".log"):
                    filepath = os.path.join(root, file)
                    
                    # Extract info from path
                    # Expected: .../action/seed_X/...
                    parts = filepath.split(os.sep)
                    try:
                        # Find 'seed_X' directory
                        seed_dir_idx = -2
                        seed_dir = parts[seed_dir_idx]
                        if not seed_dir.startswith("seed_"):
                             # Maybe nested differently? Try searching backward
                             for i in range(len(parts)-1, -1, -1):
                                 if parts[i].startswith("seed_"):
                                     seed_dir = parts[i]
                                     seed_dir_idx = i
                                     break
                        
                        seed = seed_dir.split("_")[1]
                        
                        # Action is parent of seed dir
                        action = parts[seed_dir_idx - 1]
                        
                        # Parse metrics
                        auc_roc, auc_pr = parse_log_file(filepath)
                        
                        if auc_roc is not None:
                            results.append({
                                "Model": model_name,
                                "Action": action,
                                "Seed": seed,
                                "AUC ROC": auc_roc,
                                "AUC PR": auc_pr,
                                "Lambda": "N/A", # Not varying lambda here
                                "Experiment": "Teacher vs Baseline"
                            })
                            
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                        continue

    df = pd.DataFrame(results)
    output_file = "teacher_comparison_results_new_2.csv"
    df.to_csv(output_file, index=False)
    print(f"Aggregated {len(df)} results to {output_file}")
    print(df.head())

if __name__ == "__main__":
    aggregate_results()
