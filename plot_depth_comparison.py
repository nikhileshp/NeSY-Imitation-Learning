import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
base_dir = "rdn_models/seaquest/all"
actions = ["fire", "up", "down", "left", "right", "noop"]
seeds = [1729, 42, 123, 456, 789]
depths = [1, 2, 3]
negpos_ratios = [1, 2]
grounding_penalty = "0.1"
trees = 1

for negpos in negpos_ratios:
    print(f"\nProcessing NegPos Ratio: {negpos}")
    results = []

    print("Parsing logs...")
    for depth in depths:
        dir_name = f"negpos_2_trees_{trees}_depth_{depth}_grounding_penalty_{grounding_penalty}"
        
        for action in actions:
            f1_scores = []
            auc_pr_scores = []
            
            for seed in seeds:
                # Try both naming conventions just in case
                log_filename_1 = f"action_test_infer_seed_{seed}_negpos_{negpos}.log"
                log_path_1 = os.path.join(base_dir, dir_name, action, log_filename_1)
                
                log_filename_2 = f"action_test_infer_seed_{seed}.log"
                log_path_2 = os.path.join(base_dir, dir_name, action, log_filename_2)
                
                log_path = None
                if os.path.exists(log_path_1):
                    log_path = log_path_1
                elif negpos == 2 and os.path.exists(log_path_2):
                    # Fallback only valid for negpos 2 if that was the default
                    log_path = log_path_2
                
                if not log_path:
                    # print(f"Warning: Log not found for {action} depth {depth} seed {seed}")
                    continue
                
                with open(log_path, 'r') as f:
                    content = f.read()
                    
                # Extract metrics
                f1_match = re.search(r'%\s+F1\s*[:=]\s*([\d\.]+)', content)
                auc_pr_match = re.search(r'AUC PR\s*[:=]\s*([\d\.]+)', content)
                
                if f1_match: f1_scores.append(float(f1_match.group(1)))
                if auc_pr_match: auc_pr_scores.append(float(auc_pr_match.group(1)))
            
            if f1_scores:
                results.append({
                    'Depth': depth,
                    'Action': action,
                    'F1_Mean': np.mean(f1_scores),
                    'F1_Std': np.std(f1_scores),
                    'AUC_PR_Mean': np.mean(auc_pr_scores) if auc_pr_scores else 0,
                    'AUC_PR_Std': np.std(auc_pr_scores) if auc_pr_scores else 0,
                    'Count': len(f1_scores)
                })

    df = pd.DataFrame(results)

    if df.empty:
        print(f"No results found for NegPos {negpos}!")
        continue

    print(f"\nData Summary (NegPos {negpos}):")
    print(df)

    # Plotting
    print(f"\nGenerating Plots for NegPos {negpos}...")

    # 1. F1 Score Plot
    pivot_f1_mean = df.pivot(index="Action", columns="Depth", values="F1_Mean")
    pivot_f1_std = df.pivot(index="Action", columns="Depth", values="F1_Std")

    if not pivot_f1_mean.empty:
        ax = pivot_f1_mean.plot(kind='bar', yerr=pivot_f1_std, capsize=4, figsize=(12, 6), rot=0, alpha=0.8)
        plt.title(f'F1 Score by Depth (Grounding Penalty {grounding_penalty}, Trees {trees}, NegPos {negpos})')
        plt.ylabel('F1 Score')
        plt.xlabel('Action')
        plt.ylim(0, 1)
        plt.legend(title='Depth')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        filename = f'f1_depth_comparison_negpos_{negpos}.png'
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

    # 2. AUC PR Plot
    pivot_auc_mean = df.pivot(index="Action", columns="Depth", values="AUC_PR_Mean")
    pivot_auc_std = df.pivot(index="Action", columns="Depth", values="AUC_PR_Std")

    if not pivot_auc_mean.empty:
        ax = pivot_auc_mean.plot(kind='bar', yerr=pivot_auc_std, capsize=4, figsize=(12, 6), rot=0, alpha=0.8)
        plt.title(f'AUC PR by Depth (Grounding Penalty {grounding_penalty}, Trees {trees}, NegPos {negpos})')
        plt.ylabel('AUC PR')
        plt.xlabel('Action')
        plt.ylim(0, 1)
        plt.legend(title='Depth')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        filename = f'auc_pr_depth_comparison_negpos_{negpos}.png'
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()
