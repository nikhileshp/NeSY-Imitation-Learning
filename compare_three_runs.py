import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODELS = {
    "Baseline": "rdn_models/seaquest/all/negpos_2_trees_1_depth_3_new",
    "Teacher Only": "rdn_models/seaquest/teacher_only/negpos_2_trees_1_depth_3",
    "Joint Training": "rdn_models/seaquest/joint/negpos_2_trees_1_depth_3_lambda_1.0"
}

ACTIONS = ["fire", "up", "down", "left", "right", "noop"]
SEEDS = [42]

def parse_log_file(log_path):
    """
    Parses a test_infer log file to extract AUC ROC and AUC PR.
    """
    if not os.path.exists(log_path):
        return None, None
    
    auc_pr = None
    f1_score = None
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Regex to match: "%   AUC PR    = 0.724538"
        pr_match = re.search(r"%\s+AUC PR\s+=\s+([0-9.]+)", content)
        if pr_match:
            auc_pr = float(pr_match.group(1))
            
        # Regex to match: "%   F1        = 0.699681"
        f1_match = re.search(r"%\s+F1\s+=\s+([0-9.]+)", content)
        if f1_match:
            f1_score = float(f1_match.group(1))
            
    return auc_pr, f1_score

def main():
    results = []
    
    for model_name, base_path in MODELS.items():
        print(f"Processing {model_name}...")
        for action in ACTIONS:
            for seed in SEEDS:
                # Construct path
                # Pattern: {base_path}/{action}/seed_{seed}/test_infer_seed_{seed}.log
                log_path = os.path.join(base_path, action, f"seed_{seed}", f"test_infer_seed_{seed}.log")
                
                if not os.path.exists(log_path):
                    # Check for direct log file in action dir fallback
                    log_path_fallback = os.path.join(base_path, action, f"test_infer_seed_{seed}.log")
                    if os.path.exists(log_path_fallback):
                        log_path = log_path_fallback
                    else:
                        print(f"  Warning: Log not found at {log_path}")
                        continue
                
                pr, f1 = parse_log_file(log_path)
                
                if pr is not None and f1 is not None:
                    results.append({
                        "Model": model_name,
                        "Action": action,
                        "Seed": seed,
                        "AUC_PR": pr,
                        "F1_Score": f1
                    })
                else:
                    print(f"  Warning: Could not parse metrics from {log_path}")

    if not results:
        print("No results found!")
        return

    df = pd.DataFrame(results)
    print("\nAggregated Results:")
    print(df)
    
    # Save to CSV
    csv_filename = "three_model_comparison_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved to {csv_filename}")
    
    # Visualization
    pivot_pr = df.pivot(index="Action", columns="Model", values="AUC_PR")
    
    plt.figure(figsize=(10, 6))
    
    # Plot AUC PR
    pivot_pr.plot(kind='bar', width=0.8)
    plt.title("AUC PR Comparison")
    plt.ylabel("AUC PR")
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Model")
    
    plt.tight_layout()
    plt.savefig("three_models_comparison_plot.png")
    print("Saved plot to three_models_comparison_plot.png")

if __name__ == "__main__":
    main()
