import os
import re
import matplotlib.pyplot as plt
import numpy as np

base_path = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/rdn_models/seaquest/all"
configs = {
    "0.1 (New)": "negpos_2_trees_1_depth_3_grounding_penalty_0.1_new",
    "0.1 (Old)": "negpos_2_trees_1_depth_3_grounding_penalty_0.1",
    "None": "negpos_2_trees_1_depth_3"
}
actions = ["down", "fire", "left", "noop", "right", "up"]
seed = "seed_42"

results = {}

for config_name, config_dir in configs.items():
    results[config_name] = {}
    
    # Extract F1 from eval_report
    eval_report_path = os.path.join(base_path, config_dir, f"eval_report_{seed}.txt")
    f1_scores = {}
    if os.path.exists(eval_report_path):
        with open(eval_report_path, "r") as f:
            content = f.read()
            for action in actions:
                match = re.search(rf"^\s*{action}\s+\S+\s+\S+\s+(\S+)", content, re.MULTILINE)
                if match:
                    f1_scores[action] = float(match.group(1))
    
    for action in actions:
        # Extract AUC
        auc_path = os.path.join(base_path, config_dir, action, seed, "test_AUC", "outputFromAUC_FILTERED.txt")
        auc_pr = 0.0
        if os.path.exists(auc_path):
            with open(auc_path, "r") as f:
                content = f.read()
                pr_match = re.search(r"Area Under the Curve for Precision - Recall is ([\d\.]+)", content)
                auc_pr = float(pr_match.group(1)) if pr_match else 0.0
        
        results[config_name][action] = {
            "AUC-PR": auc_pr, 
            "F1": f1_scores.get(action, 0.0)
        }

# Plotting
metrics = ["AUC-PR", "F1"]
x = np.arange(len(actions))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    for j, (config_name, config_data) in enumerate(results.items()):
        values = [config_data.get(action, {}).get(metric, 0.0) for action in actions]
        offset = (j - 1) * width
        rects = ax.bar(x + offset, values, width, label=config_name)
        
        # Add labels on top of bars
        # for rect in rects:
        #     height = rect.get_height()
        #     ax.annotate(f'{height:.2f}',
        #                 xy=(rect.get_x() + rect.get_width() / 2, height),
        #                 xytext=(0, 3),  # 3 points vertical offset
        #                 textcoords="offset points",
        #                 ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Action and Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.legend()
    ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('grounding_penalty_comparison_01.png')
print("Plot saved to grounding_penalty_comparison_01.png")

# Print table for verification
print(f"{'Config':<15} {'Action':<10} {'AUC-PR':<10} {'F1':<10}")
print("-" * 50)
for config_name in configs:
    for action in actions:
        data = results[config_name].get(action, {})
        auc_pr = data.get("AUC-PR", 0.0)
        f1 = data.get("F1", 0.0)
        print(f"{config_name:<15} {action:<10} {auc_pr:.4f}     {f1:.4f}")
