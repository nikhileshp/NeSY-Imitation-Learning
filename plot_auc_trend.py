import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
base_dir = "rdn_models/seaquest/all"
actions = ["fire", "up", "down", "left", "right", "noop"]
seeds = [1729, 42, 123, 456, 789]
negpos = 2
trees = 1
depth_list = [1, 2, 3]
config_suffix = "grounding_penalty_0.1_new"

results = []

print("="*80)
print(f"Analyzing AUC PR Trend for {config_suffix} (NegPos={negpos}, Trees={trees})")
print("="*80)

for depth in depth_list:
    dir_name = f"negpos_{negpos}_trees_{trees}_depth_{depth}_{config_suffix}"
    
    for action in actions:
        action_dir = os.path.join(base_dir, dir_name, action)
        
        if not os.path.exists(action_dir):
            continue
        
        auc_pr_scores = []
        
        for seed in seeds:
            # Try both formats or specific format based on directory structure
            log_path = os.path.join(action_dir, f"action_test_infer_seed_{seed}_negpos_{negpos}.log")
            
            if not os.path.exists(log_path):
                log_path = os.path.join(action_dir, f"action_test_infer_seed_{seed}.log")
                
            if not os.path.exists(log_path):
                continue
            
            with open(log_path, "r") as f:
                content = f.read()
            
            auc_pr_match = re.search(r"%   AUC PR    = ([\d\.]+)", content)
            
            if auc_pr_match:
                auc_pr_scores.append(float(auc_pr_match.group(1)))
        
        if auc_pr_scores:
            results.append({
                "Depth": depth,
                "Action": action,
                "AUC_PR_Mean": np.mean(auc_pr_scores),
                "AUC_PR_Std": np.std(auc_pr_scores)
            })

df = pd.DataFrame(results)

if df.empty:
    print("\nNo results found!")
else:
    # Pivot for easier plotting
    pivot_auc = df.pivot(index="Depth", columns="Action", values="AUC_PR_Mean")
    
    print("\nAUC PR Mean by Depth:")
    print(pivot_auc.to_string())
    
    # Calculate trend (Slope or simple difference)
    print("\nTrend Analysis (Depth 3 - Depth 1):")
    if 1 in pivot_auc.index and 3 in pivot_auc.index:
        diff = pivot_auc.loc[3] - pivot_auc.loc[1]
        for action, value in diff.items():
            trend = "IMPROVING" if value > 0 else "DECLINING" if value < 0 else "FLAT"
            print(f"  {action:8s}: {value:+.4f} ({trend})")
            
    # Plotting
    plt.figure(figsize=(10, 6))
    for action in pivot_auc.columns:
        plt.plot(pivot_auc.index, pivot_auc[action], marker='o', label=action, linewidth=2)
        
    plt.title(f'AUC PR Trend with Depth (GP 0.1 New, NegPos {negpos})')
    plt.ylabel('AUC PR')
    plt.xlabel('Depth')
    plt.xticks(depth_list)
    plt.legend(title='Action')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('auc_pr_trend_gp_0.1_new.png')
    print("\nSaved plot to auc_pr_trend_gp_0.1_new.png")
