import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
base_dir = "rdn_models/seaquest/all"
actions = ["fire", "up", "down", "left", "right", "noop"]
seeds = [1729, 42, 123, 456, 789]
negpos_list = [1, 2]
trees = 1
depth_list = [1, 2, 3]

results = []

print("="*80)
print(f"Comparing Grounding Penalty 0.1 vs 0.1_new (Trees={trees}, Depths={depth_list})")
print("="*80)

for depth in depth_list:
    for negpos in negpos_list:
        print(f"\nProcessing Depth {depth}, NegPos {negpos}...")
        
        configs = [
            ("GP_0.1", f"negpos_{negpos}_trees_{trees}_depth_{depth}_grounding_penalty_0.1"),
            ("GP_0.1_NEW", f"negpos_{negpos}_trees_{trees}_depth_{depth}_grounding_penalty_0.1_new")
        ]

        for config_name, dir_name in configs:
            for action in actions:
                action_dir = os.path.join(base_dir, dir_name, action)
                
                if not os.path.exists(action_dir):
                    # print(f"  Warning: Directory {action_dir} not found")
                    continue
                
                f1_scores = []
                auc_pr_scores = []
                
                for seed in seeds:
                    # Try both formats or specific format based on directory structure
                    # Based on ls output: action_test_infer_seed_{seed}_negpos_{negpos}.log
                    log_path = os.path.join(action_dir, f"action_test_infer_seed_{seed}_negpos_{negpos}.log")
                    
                    if not os.path.exists(log_path):
                        # Fallback to standard name if specific one doesn't exist
                        log_path = os.path.join(action_dir, f"action_test_infer_seed_{seed}.log")
                        
                    if not os.path.exists(log_path):
                        continue
                    
                    with open(log_path, "r") as f:
                        content = f.read()
                    
                    f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
                    auc_pr_match = re.search(r"%   AUC PR    = ([\d\.]+)", content)
                    
                    if f1_match:
                        f1_scores.append(float(f1_match.group(1)))
                    if auc_pr_match:
                        auc_pr_scores.append(float(auc_pr_match.group(1)))
                
                if f1_scores:
                    results.append({
                        "Depth": depth,
                        "NegPos": negpos,
                        "Config": config_name,
                        "Action": action,
                        "F1_Mean": np.mean(f1_scores),
                        "F1_Std": np.std(f1_scores),
                        "F1_Min": np.min(f1_scores),
                        "F1_Max": np.max(f1_scores),
                        "AUC_PR_Mean": np.mean(auc_pr_scores) if auc_pr_scores else None,
                        "AUC_PR_Std": np.std(auc_pr_scores) if auc_pr_scores else None,
                        "N_Seeds": len(f1_scores)
                    })

df = pd.DataFrame(results)

if df.empty:
    print("\nNo results found!")
else:
    # Generate plots for each Depth and NegPos
    for depth in depth_list:
        for negpos in negpos_list:
            subset_df = df[(df["Depth"] == depth) & (df["NegPos"] == negpos)]
            
            if subset_df.empty:
                print(f"\nNo data for Depth {depth}, NegPos {negpos}")
                continue
                
            print(f"\nGenerating plots for Depth {depth}, NegPos {negpos}...")
            
            # 1. F1 Score Plot
            pivot_f1_mean = subset_df.pivot_table(index="Action", columns="Config", values="F1_Mean", aggfunc='first')
            pivot_f1_std = subset_df.pivot_table(index="Action", columns="Config", values="F1_Std", aggfunc='first')
            
            if not pivot_f1_mean.empty:
                # Ensure consistent order
                cols = sorted(pivot_f1_mean.columns)
                pivot_f1_mean = pivot_f1_mean[cols]
                pivot_f1_std = pivot_f1_std[cols]
                
                ax = pivot_f1_mean.plot(kind='bar', yerr=pivot_f1_std, capsize=4, figsize=(10, 6), rot=0, alpha=0.8)
                plt.title(f'F1 Score Comparison: GP 0.1 vs 0.1_new (Depth {depth}, NegPos {negpos})')
                plt.ylabel('F1 Score')
                plt.xlabel('Action')
                plt.ylim(0, 1)
                plt.legend(title='Configuration')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                filename = f'f1_comparison_gp_0.1_vs_new_depth_{depth}_negpos_{negpos}.png'
                plt.savefig(filename)
                print(f"  Saved {filename}")
                plt.close()

            # 2. AUC PR Plot
            pivot_auc_mean = subset_df.pivot_table(index="Action", columns="Config", values="AUC_PR_Mean", aggfunc='first')
            pivot_auc_std = subset_df.pivot_table(index="Action", columns="Config", values="AUC_PR_Std", aggfunc='first')
            
            if not pivot_auc_mean.empty and not pivot_auc_mean.isnull().all().all():
                cols = sorted(pivot_auc_mean.columns)
                pivot_auc_mean = pivot_auc_mean[cols]
                pivot_auc_std = pivot_auc_std[cols]
                
                ax = pivot_auc_mean.plot(kind='bar', yerr=pivot_auc_std, capsize=4, figsize=(10, 6), rot=0, alpha=0.8)
                plt.title(f'AUC PR Comparison: GP 0.1 vs 0.1_new (Depth {depth}, NegPos {negpos})')
                plt.ylabel('AUC PR')
                plt.xlabel('Action')
                plt.ylim(0, 1)
                plt.legend(title='Configuration')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                filename = f'auc_pr_comparison_gp_0.1_vs_new_depth_{depth}_negpos_{negpos}.png'
                plt.savefig(filename)
                print(f"  Saved {filename}")
                plt.close()

    # Print Summary Tables
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    for depth in depth_list:
        for negpos in negpos_list:
            subset_df = df[(df["Depth"] == depth) & (df["NegPos"] == negpos)]
            if subset_df.empty: continue
            
            print(f"\nDepth {depth}, NegPos {negpos}:")
            print("-" * 40)
            
            pivot_f1 = subset_df.pivot_table(index="Action", columns="Config", values="F1_Mean", aggfunc='first')
            if 'GP_0.1_NEW' in pivot_f1.columns and 'GP_0.1' in pivot_f1.columns:
                pivot_f1['Diff'] = pivot_f1['GP_0.1_NEW'] - pivot_f1['GP_0.1']
            
            print("F1 Mean:")
            print(pivot_f1.to_string())
            
            pivot_auc = subset_df.pivot_table(index="Action", columns="Config", values="AUC_PR_Mean", aggfunc='first')
            if 'GP_0.1_NEW' in pivot_auc.columns and 'GP_0.1' in pivot_auc.columns:
                pivot_auc['Diff'] = pivot_auc['GP_0.1_NEW'] - pivot_auc['GP_0.1']
                
            print("\nAUC PR Mean:")
            print(pivot_auc.to_string())
