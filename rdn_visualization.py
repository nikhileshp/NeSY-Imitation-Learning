#!/usr/bin/env python3
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_weighted_f1_from_eval_report(file_path):
    """Extract weighted F1 score from eval_report.txt file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for the weighted avg line
        weighted_match = re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+', content)
        if weighted_match:
            return float(weighted_match.group(3))  # F1 score is the 3rd number
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_metrics_from_log_file(file_path):
    """Extract AUC-PR and F1 scores from log file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract AUC PR
        auc_pr_match = re.search(r'%\s+AUC PR\s+=\s+([\d.]+)', content)
        auc_pr = float(auc_pr_match.group(1)) if auc_pr_match else None
        
        # Extract F1
        f1_match = re.search(r'%\s+F1\s+=\s+([\d.]+)', content)
        f1 = float(f1_match.group(1)) if f1_match else None
        
        return auc_pr, f1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def parse_model_info(path):
    """Parse model type and depth from path"""
    path_parts = path.split('/')
    
    # Find the model folder name
    model_folder = None
    for part in path_parts:
        if 'depth_' in part:
            model_folder = part
            break
    
    if not model_folder:
        return None, None
    
    # Extract depth
    depth_match = re.search(r'depth_(\d+)', model_folder)
    depth = int(depth_match.group(1)) if depth_match else None
    
    # Determine model type
    if 'weighted_1object' in model_folder:
        model_type = 'weighted_1object'
    elif 'w_abstraction' in model_folder:
        model_type = 'unweighted_w_abstraction'
    elif 'unweighted' in model_folder:
        model_type = 'unweighted'
    else:
        model_type = 'unknown'
    
    return model_type, depth

def collect_data():
    """Collect all metrics from the RDN models"""
    base_dir = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/rdn_models/seaquest"
    
    # Data structures to store results
    eval_data = []  # For weighted F1 from eval_report
    log_data = []   # For AUC-PR and F1 from log files
    
    # Find all eval_report files
    for root, dirs, files in os.walk(base_dir):
        if 'eval_report.txt' in files:
            eval_path = os.path.join(root, 'eval_report.txt')
            model_type, depth = parse_model_info(root)
            
            if model_type and depth is not None:
                weighted_f1 = extract_weighted_f1_from_eval_report(eval_path)
                if weighted_f1 is not None:
                    eval_data.append({
                        'model_type': model_type,
                        'depth': depth,
                        'weighted_f1': weighted_f1
                    })
        
        # Find all log files in action folders
        for file in files:
            if file.endswith('_infer.log'):
                log_path = os.path.join(root, file)
                
                # Get model info from parent directory
                parent_dir = os.path.dirname(root)
                model_type, depth = parse_model_info(parent_dir)
                
                if model_type and depth is not None:
                    auc_pr, f1 = extract_metrics_from_log_file(log_path)
                    action = file.replace('_infer.log', '')
                    
                    if auc_pr is not None or f1 is not None:
                        log_data.append({
                            'model_type': model_type,
                            'depth': depth,
                            'action': action,
                            'auc_pr': auc_pr,
                            'f1': f1
                        })
    
    return pd.DataFrame(eval_data), pd.DataFrame(log_data)

def create_visualizations(eval_df, log_df):
    """Create visualizations comparing model performance"""
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('RDN Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Define colors for each model type
    colors = {
        'unweighted': '#1f77b4',
        'unweighted_w_abstraction': '#ff7f0e', 
        'weighted_1object': '#2ca02c'
    }
    
    # Define markers for each action
    action_markers = {
        'down': 'v', 'up': '^', 'left': '<', 'right': '>', 
        'fire': 's', 'noop': 'o'
    }
    
    # Plot 1: Weighted F1 vs Depth (from eval_report)
    ax1 = axes[0, 0]
    for model_type in eval_df['model_type'].unique():
        model_data = eval_df[eval_df['model_type'] == model_type].sort_values('depth')
        ax1.plot(model_data['depth'], model_data['weighted_f1'], 
                marker='o', linewidth=2, markersize=8,
                color=colors.get(model_type, 'gray'),
                label=model_type.replace('_', ' ').title())
    
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Weighted F1 Score')
    ax1.set_title('Weighted F1 Score vs Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual Action AUC-PR vs Depth
    ax2 = axes[0, 1]
    for action in log_df['action'].unique():
        action_data = log_df[log_df['action'] == action]
        for model_type in action_data['model_type'].unique():
            model_action_data = action_data[action_data['model_type'] == model_type].sort_values('depth')
            if not model_action_data.empty:
                ax2.plot(model_action_data['depth'], model_action_data['auc_pr'], 
                        marker=action_markers.get(action, 'o'), linewidth=1.5, markersize=6,
                        color=colors.get(model_type, 'gray'), alpha=0.7,
                        label=f"{model_type.replace('_', ' ').title()} - {action}")
    
    ax2.set_xlabel('Depth')
    ax2.set_ylabel('AUC-PR')
    ax2.set_title('Individual Action AUC-PR vs Depth')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual Action F1 vs Depth
    ax3 = axes[0, 2]
    for action in log_df['action'].unique():
        action_data = log_df[log_df['action'] == action]
        for model_type in action_data['model_type'].unique():
            model_action_data = action_data[action_data['model_type'] == model_type].sort_values('depth')
            if not model_action_data.empty:
                ax3.plot(model_action_data['depth'], model_action_data['f1'], 
                        marker=action_markers.get(action, 'o'), linewidth=1.5, markersize=6,
                        color=colors.get(model_type, 'gray'), alpha=0.7,
                        label=f"{model_type.replace('_', ' ').title()} - {action}")
    
    ax3.set_xlabel('Depth')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Individual Action F1 vs Depth')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average AUC-PR vs Depth (aggregated)
    ax4 = axes[1, 0]
    auc_pr_avg = log_df.groupby(['model_type', 'depth'])['auc_pr'].mean().reset_index()
    
    for model_type in auc_pr_avg['model_type'].unique():
        model_data = auc_pr_avg[auc_pr_avg['model_type'] == model_type].sort_values('depth')
        ax4.plot(model_data['depth'], model_data['auc_pr'], 
                marker='s', linewidth=2, markersize=8,
                color=colors.get(model_type, 'gray'),
                label=model_type.replace('_', ' ').title())
    
    ax4.set_xlabel('Depth')
    ax4.set_ylabel('Average AUC-PR')
    ax4.set_title('Average AUC-PR vs Depth')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Average F1 vs Depth (aggregated)
    ax5 = axes[1, 1]
    f1_avg = log_df.groupby(['model_type', 'depth'])['f1'].mean().reset_index()
    
    for model_type in f1_avg['model_type'].unique():
        model_data = f1_avg[f1_avg['model_type'] == model_type].sort_values('depth')
        ax5.plot(model_data['depth'], model_data['f1'], 
                marker='^', linewidth=2, markersize=8,
                color=colors.get(model_type, 'gray'),
                label=model_type.replace('_', ' ').title())
    
    ax5.set_xlabel('Depth')
    ax5.set_ylabel('Average F1 Score')
    ax5.set_title('Average F1 Score vs Depth')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Combined comparison at different depths
    ax6 = axes[1, 2]
    
    # Create a grouped bar chart showing all metrics at key depths
    depths_to_show = sorted(eval_df['depth'].unique())[:6]  # Show first 6 depths
    x_pos = np.arange(len(depths_to_show))
    width = 0.25
    
    for i, model_type in enumerate(['unweighted', 'unweighted_w_abstraction', 'weighted_1object']):
        if model_type in eval_df['model_type'].values:
            model_f1_scores = []
            for depth in depths_to_show:
                scores = eval_df[(eval_df['model_type'] == model_type) & 
                               (eval_df['depth'] == depth)]['weighted_f1'].values
                model_f1_scores.append(scores[0] if len(scores) > 0 else 0)
            
            ax6.bar(x_pos + i * width, model_f1_scores, width,
                   color=colors.get(model_type, 'gray'),
                   label=model_type.replace('_', ' ').title())
    
    ax6.set_xlabel('Depth')
    ax6.set_ylabel('Weighted F1 Score')
    ax6.set_title('Model Comparison by Depth')
    ax6.set_xticks(x_pos + width)
    ax6.set_xticklabels(depths_to_show)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rdn_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_action_plots(log_df):
    """Create separate plots for each action's performance"""
    actions = sorted(log_df['action'].unique())
    n_actions = len(actions)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Action Performance by Model Type', fontsize=16, fontweight='bold')
    
    colors = {
        'unweighted': '#1f77b4',
        'unweighted_w_abstraction': '#ff7f0e', 
        'weighted_1object': '#2ca02c'
    }
    
    for i, action in enumerate(actions):
        if i < 6:  # Only plot first 6 actions
            ax = axes[i//3, i%3]
            action_data = log_df[log_df['action'] == action]
            
            # Plot AUC-PR
            for model_type in action_data['model_type'].unique():
                model_data = action_data[action_data['model_type'] == model_type].sort_values('depth')
                if not model_data.empty:
                    ax.plot(model_data['depth'], model_data['auc_pr'], 
                           marker='o', linewidth=2, markersize=6,
                           color=colors.get(model_type, 'gray'),
                           label=model_type.replace('_', ' ').title())
            
            ax.set_xlabel('Depth')
            ax.set_ylabel('AUC-PR')
            ax.set_title(f'Action: {action.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('individual_action_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(eval_df, log_df):
    """Print summary statistics"""
    print("\n=== SUMMARY STATISTICS ===\n")
    
    print("Weighted F1 Scores by Model Type:")
    print(eval_df.groupby('model_type')['weighted_f1'].agg(['count', 'mean', 'std', 'min', 'max']).round(4))
    
    print("\nAUC-PR Scores by Model Type:")
    print(log_df.groupby('model_type')['auc_pr'].agg(['count', 'mean', 'std', 'min', 'max']).round(4))
    
    print("\nOne-Class F1 Scores by Model Type:")  
    print(log_df.groupby('model_type')['f1'].agg(['count', 'mean', 'std', 'min', 'max']).round(4))
    
    print("\n=== INDIVIDUAL ACTION PERFORMANCE ===\n")
    
    print("AUC-PR by Action and Model Type:")
    action_auc_pivot = log_df.pivot_table(values='auc_pr', index='action', columns='model_type', aggfunc='mean')
    print(action_auc_pivot.round(4))
    
    print("\nF1 by Action and Model Type:")
    action_f1_pivot = log_df.pivot_table(values='f1', index='action', columns='model_type', aggfunc='mean')
    print(action_f1_pivot.round(4))

if __name__ == "__main__":
    print("Collecting RDN model data...")
    eval_df, log_df = collect_data()
    
    print(f"Found {len(eval_df)} eval_report entries and {len(log_df)} log file entries")
    
    if not eval_df.empty or not log_df.empty:
        print("\nCreating main comparison visualization...")
        create_visualizations(eval_df, log_df)
        
        print("\nCreating individual action plots...")
        create_individual_action_plots(log_df)
        
        print_summary_statistics(eval_df, log_df)
        print("\nVisualizations saved as:")
        print("  - rdn_model_comparison.png (main comparison)")
        print("  - individual_action_performance.png (individual actions)")
    else:
        print("No data found to visualize!")
