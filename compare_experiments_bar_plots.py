import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

def get_metrics(model_dir, action, seed="seed_42"):
    f1 = 0.0
    auc_pr = 0.0
    
    # Construct path to the log file
    # Structure: model_dir/action/seed/test_infer_seed.log OR test_infer.log
    log_path = os.path.join(model_dir, action, seed, f"test_infer_{seed}.log")
    
    if not os.path.exists(log_path):
        log_path = os.path.join(model_dir, action, seed, "test_infer.log")
    
    if os.path.exists(log_path):
        print(f"Reading log: {log_path}")
        with open(log_path, 'r') as f:
            content = f.read()
            
            # Extract metrics using regex
            # RDN Pattern: %   AUC PR    = 0.776094
            # MLP Pattern: AUC PR:    0.8937
            
            pr_match = re.search(r"(?:%|)\s*AUC PR\s*(?:=|:)\s*([\d\.]+)", content)
            f1_match = re.search(r"(?:%|)\s*F1\s*(?:=|:)\s*([\d\.]+)", content)
            
            if pr_match: auc_pr = float(pr_match.group(1))
            if f1_match: f1 = float(f1_match.group(1))
            
    return f1, auc_pr

def main():
    parser = argparse.ArgumentParser(description="Compare RDN models using combined bar plots.")
    parser.add_argument("directories", nargs='+', help="Paths to model directories to compare")
    args = parser.parse_args()
    
    actions = ["down", "fire", "left", "noop", "right", "up"]
    seed = "seed_42"
    
    output_dir = "plots/comparison_bar_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    # Get model names
    model_names = []
    for d in args.directories:
        name = os.path.basename(d.rstrip('/'))
        if name in model_names:
            name = f"{name}_{model_names.count(name)}"
        model_names.append(name)
        
    # Collect data
    # auc_pr_data[model_idx] = [score_action1, score_action2, ...]
    auc_pr_data = [[] for _ in range(len(model_names))]
    
    for action in actions:
        for i, model_dir in enumerate(args.directories):
            _, auc_pr = get_metrics(model_dir, action, seed)
            auc_pr_data[i].append(auc_pr)
            
    # Plotting
    x = np.arange(len(actions))
    # Width of each bar
    total_width = 0.8
    bar_width = total_width / len(model_names)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors - use a qualitative colormap
    # tab10 has 10 distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot AUC-PR Scores
    for i, name in enumerate(model_names):
        offset = (i - len(model_names)/2) * bar_width + bar_width/2
        rects = ax.bar(x + offset, auc_pr_data[i], bar_width, label=name, color=colors[i % 10])
        ax.bar_label(rects, padding=3, fmt='%.2f', rotation=90, fontsize=8)
        
    ax.set_ylabel('AUC-PR Score')
    ax.set_title('AUC-PR Score Comparison by Action and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "comparison_combined.png")
    plt.savefig(output_file)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    main()
