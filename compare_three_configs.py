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

results = []

print("="*80)
print(f"Comparing Three Configurations (NegPos={negpos}, Trees={trees}, Depths={depth_list})")
print("="*80)

for depth in depth_list:
    print(f"\nProcessing Depth {depth}...")
    
    configs = [
        ("GP_0.1", f"negpos_{negpos}_trees_{trees}_depth_{depth}_grounding_penalty_0.1"),
        ("GP_0.1_NEW", f"negpos_{negpos}_trees_{trees}_depth_{depth}_grounding_penalty_0.1_new"),
        ("NO_GP", f"negpos_{negpos}_trees_{trees}_depth_{depth}")
    ]

    for config_name, dir_name in configs:
        for action in actions:
            action_dir = os.path.join(base_dir, dir_name, action)
            
            if not os.path.exists(action_dir):
                continue
            
            f1_scores = []
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
                
                f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
                auc_pr_match = re.search(r"%   AUC PR    = ([\d\.]+)", content)
                
                if f1_match:
                    f1_scores.append(float(f1_match.group(1)))
                if auc_pr_match:
                    auc_pr_scores.append(float(auc_pr_match.group(1)))
            
            if f1_scores:
                results.append({
                    "Depth": depth,
                    "Config": config_name,
                    "Action": action,
                    "F1_Mean": np.mean(f1_scores),
                    "F1_Std": np.std(f1_scores),
                    "AUC_PR_Mean": np.mean(auc_pr_scores) if auc_pr_scores else None,
                    "AUC_PR_Std": np.std(auc_pr_scores) if auc_pr_scores else None,
                    "N_Seeds": len(f1_scores)
                })

df = pd.DataFrame(results)

if df.empty:
    print("\nNo results found!")
else:
    # 1. Bar Plots for F1 and AUC PR (Grouped by Depth)
    for depth in depth_list:
        subset_df = df[df["Depth"] == depth]
        if subset_df.empty: continue
        
        print(f"\nGenerating bar plots for Depth {depth}...")
        
        # F1 Plot
        pivot_f1_mean = subset_df.pivot_table(index="Action", columns="Config", values="F1_Mean", aggfunc='first')
        pivot_f1_std = subset_df.pivot_table(index="Action", columns="Config", values="F1_Std", aggfunc='first')
        
        if not pivot_f1_mean.empty:
            # Ensure consistent order: GP_0.1, GP_0.1_NEW, NO_GP
            cols = [c for c in ["GP_0.1", "GP_0.1_NEW", "NO_GP"] if c in pivot_f1_mean.columns]
            pivot_f1_mean = pivot_f1_mean[cols]
            pivot_f1_std = pivot_f1_std[cols]
            
            ax = pivot_f1_mean.plot(kind='bar', yerr=pivot_f1_std, capsize=4, figsize=(12, 6), rot=0, alpha=0.8)
            plt.title(f'F1 Score Comparison (Depth {depth})')
            plt.ylabel('F1 Score')
            plt.xlabel('Action')
            plt.ylim(0, 1)
            plt.legend(title='Configuration')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'f1_comparison_three_configs_depth_{depth}.png')
            plt.close()

        # AUC PR Plot
        pivot_auc_mean = subset_df.pivot_table(index="Action", columns="Config", values="AUC_PR_Mean", aggfunc='first')
        pivot_auc_std = subset_df.pivot_table(index="Action", columns="Config", values="AUC_PR_Std", aggfunc='first')
        
        if not pivot_auc_mean.empty:
            cols = [c for c in ["GP_0.1", "GP_0.1_NEW", "NO_GP"] if c in pivot_auc_mean.columns]
            pivot_auc_mean = pivot_auc_mean[cols]
            pivot_auc_std = pivot_auc_std[cols]
            
            ax = pivot_auc_mean.plot(kind='bar', yerr=pivot_auc_std, capsize=4, figsize=(12, 6), rot=0, alpha=0.8)
            plt.title(f'AUC PR Comparison (Depth {depth})')
            plt.ylabel('AUC PR')
            plt.xlabel('Action')
            plt.ylim(0, 1)
            plt.legend(title='Configuration')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'auc_pr_comparison_three_configs_depth_{depth}.png')
            plt.close()

    # 2. Line Plot for AUC PR vs Depth (Aggregated or per action? User asked for "how aucpr changes for the 3 models with depth")
    # I'll create one plot per action to show the trend clearly, or an average if preferred. 
    # Let's do one plot with subplots for each action.
    
    print("\nGenerating AUC PR trend line plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, action in enumerate(actions):
        ax = axes[i]
        action_df = df[df["Action"] == action]
        
        if action_df.empty:
            ax.set_title(f"{action} (No Data)")
            continue
            
        pivot_trend = action_df.pivot(index="Depth", columns="Config", values="AUC_PR_Mean")
        
        # Ensure consistent order/colors
        colors = {"GP_0.1": "blue", "GP_0.1_NEW": "orange", "NO_GP": "green"}
        
        for config in ["GP_0.1", "GP_0.1_NEW", "NO_GP"]:
            if config in pivot_trend.columns:
                ax.plot(pivot_trend.index, pivot_trend[config], marker='o', label=config, color=colors.get(config, "black"), linewidth=2)
        
        ax.set_title(f"AUC PR Trend: {action.upper()}")
        ax.set_xlabel("Depth")
        ax.set_ylabel("AUC PR")
        ax.set_ylim(0, 1)
        ax.set_xticks(depth_list)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0: # Legend only on first plot to avoid clutter, or maybe outside?
            ax.legend()
            
    plt.tight_layout()
    plt.savefig('auc_pr_trend_three_configs.png')
    plt.close()
    
    print("\nSaved plots:")
    print("- f1_comparison_three_configs_depth_*.png")
    print("- auc_pr_comparison_three_configs_depth_*.png")
    print("- auc_pr_trend_three_configs.png")

    # Summary Tables
    print("\n" + "="*80)
    print("Summary Statistics (AUC PR Mean)")
    print("="*80)
    
    for depth in depth_list:
        print(f"\nDepth {depth}:")
        subset_df = df[df["Depth"] == depth]
        pivot_auc = subset_df.pivot_table(index="Action", columns="Config", values="AUC_PR_Mean", aggfunc='first')
        cols = [c for c in ["GP_0.1", "GP_0.1_NEW", "NO_GP"] if c in pivot_auc.columns]
        print(pivot_auc[cols].to_string())
