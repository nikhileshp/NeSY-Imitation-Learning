import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
ACTIONS = ["fire", "up", "down", "left", "right", "noop"]
SEEDS = [42, 123, 456, 789, 1729]
DEPTHS = [1, 2, 3]
BASE_DIR = "rdn_models/seaquest/all"
PLOTS_DIR = "experiments/plots"

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_log(file_path):
    """Parses a log file to extract AUC PR, AUC ROC, and F1 scores."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "r") as f:
        content = f.read()
        
    auc_pr_match = re.search(r"AUC PR\s*[=:]\s*([0-9.]+)", content)
    auc_roc_match = re.search(r"AUC ROC\s*[=:]\s*([0-9.]+)", content)
    f1_match = re.search(r"F1\s*[=:]\s*([0-9.]+)", content)
    
    if auc_pr_match and auc_roc_match and f1_match:
        return {
            "auc_pr": float(auc_pr_match.group(1)),
            "auc_roc": float(auc_roc_match.group(1)),
            "f1": float(f1_match.group(1))
        }
    return None

def collect_data():
    """Collects data from all log files."""
    data = []
    
    # RDN Configurations
    for depth in DEPTHS:
        for penalty in [False, True]:
            config_name = f"negpos_2_trees_1_depth_{depth}"
            if penalty:
                config_name += "_grounding_penalty_0.1_new" # Based on directory listing
            
            # Check if directory exists (some might be named differently)
            # In step 5: negpos_2_trees_1_depth_3_grounding_penalty_0.1_new
            # But depth 1 is: negpos_2_trees_1_depth_1_grounding_penalty_0.1_new
            # And depth 2 is: negpos_2_trees_1_depth_2_grounding_penalty_0.1_new
            # Standard config: negpos_2_trees_1_depth_{depth}
            
            # Handle potential directory naming variations if needed, but based on ls output they seem consistent
            
            for action in ACTIONS:
                for seed in SEEDS:
                    seed_dir = os.path.join(BASE_DIR, config_name, action, f"seed_{seed}")
                    
                    # Train Inference
                    train_log = os.path.join(seed_dir, "train_infer.log")
                    metrics = parse_log(train_log)
                    if metrics:
                        data.append({
                            "Model": "RDN",
                            "Depth": depth,
                            "Penalty": penalty,
                            "Action": action,
                            "Seed": seed,
                            "Dataset": "Train",
                            **metrics
                        })
                        
                    # Test Inference (Standard)
                    test_log = os.path.join(seed_dir, f"test_infer_seed_{seed}.log")
                    metrics = parse_log(test_log)
                    if metrics:
                        data.append({
                            "Model": "RDN",
                            "Depth": depth,
                            "Penalty": penalty,
                            "Action": action,
                            "Seed": seed,
                            "Dataset": "Test (Infer)",
                            **metrics
                        })

                    # Test All (New Runs / Full Test)
                    # Based on file listing: test_infer_all_seed_{seed}.log
                    test_all_log = os.path.join(seed_dir, f"test_infer_all_seed_{seed}.log")
                    metrics = parse_log(test_all_log)
                    if metrics:
                        data.append({
                            "Model": "RDN",
                            "Depth": depth,
                            "Penalty": penalty,
                            "Action": action,
                            "Seed": seed,
                            "Dataset": "Test (All)",
                            **metrics
                        })

    # MLP Configuration
    mlp_config = "negpos_2_mlp_64_32_bc"
    for action in ACTIONS:
        for seed in SEEDS:
            # MLP structure might be different. Assuming standard structure based on previous scripts.
            # Previous script used: negpos_{ratio}_mlp_64_32_bc/{action}/seed_{seed}/test_infer.log
            # Let's assume test_infer.log exists.
            
            seed_dir = os.path.join(BASE_DIR, mlp_config, action, f"seed_{seed}")
            test_log = os.path.join(seed_dir, "test_infer.log")
            
            metrics = parse_log(test_log)
            if metrics:
                data.append({
                    "Model": "MLP",
                    "Depth": "N/A",
                    "Penalty": "N/A",
                    "Action": action,
                    "Seed": seed,
                    "Dataset": "Test (Infer)",
                    **metrics
                })

    return pd.DataFrame(data)

def generate_tables(df):
    """Generates and prints summary tables."""
    
    # 1. RDN Depth Comparison (Test Infer, No Penalty)
    print("\n" + "="*80)
    print("RDN Performance by Depth (Test Inference, No Penalty)")
    print("="*80)
    rdn_depth = df[(df["Model"] == "RDN") & (df["Penalty"] == False) & (df["Dataset"] == "Test (Infer)")]
    if not rdn_depth.empty:
        summary = rdn_depth.groupby(["Action", "Depth"])["auc_pr"].agg(["mean", "std"]).reset_index()
        print(summary.pivot(index="Action", columns="Depth", values="mean"))
    
    # 2. Dataset Comparison (Depth 3, No Penalty)
    print("\n" + "="*80)
    print("Dataset Comparison (RDN Depth 3, No Penalty)")
    print("="*80)
    rdn_d3 = df[(df["Model"] == "RDN") & (df["Depth"] == 3) & (df["Penalty"] == False)]
    if not rdn_d3.empty:
        summary = rdn_d3.groupby(["Action", "Dataset"])["auc_pr"].agg(["mean", "std"]).reset_index()
        print(summary.pivot(index="Action", columns="Dataset", values="mean"))

    # 3. Penalty Comparison (All Depths, Test Infer)
    print("\n" + "="*80)
    print("Grounding Penalty Comparison (RDN All Depths, Test Inference)")
    print("="*80)
    rdn_pen = df[(df["Model"] == "RDN") & (df["Dataset"] == "Test (Infer)")]
    if not rdn_pen.empty:
        # Create a combined column for Depth + Penalty status
        rdn_pen["Config"] = rdn_pen.apply(lambda x: f"D{x['Depth']} {'(GP)' if x['Penalty'] else ''}", axis=1)
        summary = rdn_pen.groupby(["Action", "Config"])["auc_pr"].agg(["mean"]).reset_index()
        print(summary.pivot(index="Action", columns="Config", values="mean"))

    # 4. RDN vs MLP (Best RDN Depth vs MLP)
    # Assuming Depth 3 is best for now, or we can find max.
    print("\n" + "="*80)
    print("RDN (Depth 3) vs MLP (Test Inference)")
    print("="*80)
    comparison = df[((df["Model"] == "RDN") & (df["Depth"] == 3) & (df["Penalty"] == False)) | (df["Model"] == "MLP")]
    comparison = comparison[comparison["Dataset"] == "Test (Infer)"]
    if not comparison.empty:
        summary = comparison.groupby(["Action", "Model"])["auc_pr"].agg(["mean", "std"]).reset_index()
        print(summary.pivot(index="Action", columns="Model", values="mean"))

def plot_comparisons(df):
    """Generates bar charts for comparisons."""
    
    # Filter for Test Inference
    df_test = df[df["Dataset"] == "Test (Infer)"]
    
    for action in ACTIONS:
        plt.figure(figsize=(10, 6))
        
        # Data for this action
        action_data = df_test[df_test["Action"] == action]
        
        if action_data.empty:
            continue
            
        # Group by Model/Config
        # We want to show: MLP, RDN D1, RDN D2, RDN D3, RDN D3+Penalty
        
        configs = []
        means = []
        stds = []
        
        # MLP
        mlp_data = action_data[action_data["Model"] == "MLP"]
        if not mlp_data.empty:
            configs.append("MLP")
            means.append(mlp_data["auc_pr"].mean())
            stds.append(mlp_data["auc_pr"].std())
            
        # RDN Depths (No Penalty and With Penalty)
        for d in DEPTHS:
            # No Penalty
            rdn_data = action_data[(action_data["Model"] == "RDN") & (action_data["Depth"] == d) & (action_data["Penalty"] == False)]
            if not rdn_data.empty:
                configs.append(f"RDN D{d}")
                means.append(rdn_data["auc_pr"].mean())
                stds.append(rdn_data["auc_pr"].std())
            
            # With Penalty
            rdn_pen = action_data[(action_data["Model"] == "RDN") & (action_data["Depth"] == d) & (action_data["Penalty"] == True)]
            if not rdn_pen.empty:
                configs.append(f"RDN D{d} (GP)")
                means.append(rdn_pen["auc_pr"].mean())
                stds.append(rdn_pen["auc_pr"].std())
            
        # Plot
        plt.bar(configs, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title(f"AUC-PR Comparison for Action: {action}")
        plt.ylabel("AUC-PR")
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(PLOTS_DIR, f"comparison_{action}.png"))
        plt.close()

if __name__ == "__main__":
    df = collect_data()
    if not df.empty:
        generate_tables(df)
        plot_comparisons(df)
        print(f"\nPlots saved to {PLOTS_DIR}")
        
        # Save full CSV for reference
        df.to_csv(os.path.join(PLOTS_DIR, "rdn_results.csv"), index=False)
    else:
        print("No data collected.")
