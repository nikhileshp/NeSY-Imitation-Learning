import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# Configuration
ACTIONS = ["fire", "up", "down", "left", "right", "noop"]
SEEDS = [42, 123, 456, 789, 1729]
RATIOS = [2.0, 1.0]
LOG_DIR = "rdn_models/seaquest/all"

# RDN Results (from walkthrough.md - Depth 3 NO_GP as baseline for best RDN)
# Action | AUC PR
RDN_RESULTS = {
    "down": 0.5993,
    "fire": 0.7147,
    "left": 0.7951,
    "noop": 0.8900,
    "right": 0.8121,
    "up": 0.5626
}

def parse_logs():
    results = defaultdict(lambda: defaultdict(list))
    
    for action in ACTIONS:
        for seed in SEEDS:
            for ratio in RATIOS:
                # Structure: rdn_models/seaquest/all/negpos_{NEGPOS}_mlp_64_32_bc/{action}/seed_{seed}/test_infer.log
                log_file = os.path.join(LOG_DIR, f"negpos_{int(ratio)}_mlp_64_32_bc", action, f"seed_{seed}", "test_infer.log")
                
                if not os.path.exists(log_file):
                    # print(f"Warning: Log file not found: {log_file}")
                    continue
                
                with open(log_file, "r") as f:
                    content = f.read()
                    
                # Extract metrics
                auc_pr_match = re.search(r"AUC PR:\s+([0-9.]+)", content)
                f1_match = re.search(r"F1:\s+([0-9.]+)", content)
                auc_roc_match = re.search(r"AUC ROC:\s+([0-9.]+)", content)
                
                if auc_pr_match and f1_match:
                    results[action][ratio].append({
                        "auc_pr": float(auc_pr_match.group(1)),
                        "f1": float(f1_match.group(1)),
                        "auc_roc": float(auc_roc_match.group(1)) if auc_roc_match else 0.0
                    })
                    
    return results

def print_comparison(results):
    print("\n" + "="*80)
    print(f"{'Action':<10} {'Ratio':<6} {'Cloning AUC PR (Mean ± Std)':<30} {'RDN (Best) AUC PR':<20} {'Diff':<10}")
    print("-" * 80)
    
    for action in ACTIONS:
        for ratio in RATIOS:
            metrics = results[action][ratio]
            if not metrics:
                continue
                
            auc_prs = [m["auc_pr"] for m in metrics]
            mean_auc = np.mean(auc_prs)
            std_auc = np.std(auc_prs)
            
            rdn_val = RDN_RESULTS.get(action, 0.0)
            diff = mean_auc - rdn_val
            
            print(f"{action:<10} {ratio:<6} {mean_auc:.4f} ± {std_auc:.4f}{'':<12} {rdn_val:.4f}{'':<16} {diff:+.4f}")

    print("\n" + "="*80)
    print("Detailed Cloning Results (Mean over seeds)")
    print(f"{'Action':<10} {'Ratio':<6} {'AUC PR':<10} {'AUC ROC':<10} {'F1':<10}")
    print("-" * 80)
    
    for action in ACTIONS:
        for ratio in RATIOS:
            metrics = results[action][ratio]
            if not metrics:
                continue
                
            mean_auc_pr = np.mean([m["auc_pr"] for m in metrics])
            mean_auc_roc = np.mean([m["auc_roc"] for m in metrics])
            mean_f1 = np.mean([m["f1"] for m in metrics])
            
            print(f"{action:<10} {ratio:<6} {mean_auc_pr:.4f}     {mean_auc_roc:.4f}      {mean_f1:.4f}")

if __name__ == "__main__":
    data = parse_logs()
    print_comparison(data)
