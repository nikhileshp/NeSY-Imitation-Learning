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
    """Create main comparison visualization with weighted F1 and average AUC-PR"""
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle('RDN Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Define colors for each model type
    colors = {
        'unweighted': '#1f77b4',
        'unweighted_w_abstraction': '#ff7f0e', 
        'weighted_1object': '#2ca02c'
    }
    
    # Plot 1: Weighted F1 vs Depth (from eval_report)
    ax1 = axes
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
    ax1.set_ylim(0, 1)  # Set y-axis limits
    ax1.grid(True, alpha=0.3)
    
    # # Plot 2: Average AUC-PR vs Depth (aggregated)
    # ax2 = axes[1]
    # auc_pr_avg = log_df.groupby(['model_type', 'depth'])['auc_pr'].mean().reset_index()
    
    # for model_type in auc_pr_avg['model_type'].unique():
    #     model_data = auc_pr_avg[auc_pr_avg['model_type'] == model_type].sort_values('depth')
    #     ax2.plot(model_data['depth'], model_data['auc_pr'], 
    #             marker='s', linewidth=2, markersize=8,
    #             color=colors.get(model_type, 'gray'),
    #             label=model_type.replace('_', ' ').title())
    
    # ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Average AUC-PR')
    # ax2.set_ylim(0, 1)  # Set y-axis limits
    # ax2.set_title('Average AUC-PR vs Depth')
    # ax2.legend()
    # # Set y-axis limits
    # ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rdn_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_action_f1_plots(log_df):
    """Create separate F1 plots for each action"""
    actions = sorted(log_df['action'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Action F1 Performance by Model Type', fontsize=16, fontweight='bold')
    
    colors = {
        'unweighted': '#1f77b4',
        'unweighted_w_abstraction': '#ff7f0e', 
        'weighted_1object': '#2ca02c'
    }
    
    # Plot F1 for each action
    for i, action in enumerate(actions):
        if i < 6:  # Only plot first 6 actions
            ax = axes[i//3, i%3]
            action_data = log_df[log_df['action'] == action]
            
            # Plot F1
            for model_type in action_data['model_type'].unique():
                model_data = action_data[action_data['model_type'] == model_type].sort_values('depth')
                if not model_data.empty:
                    ax.plot(model_data['depth'], model_data['f1'], 
                           marker='^', linewidth=2, markersize=6,
                           color=colors.get(model_type, 'gray'),
                           label=model_type.replace('_', ' ').title())
            
            ax.set_xlabel('Depth')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'{action.upper()} - F1')
            ax.set_ylim(0, 1)  # Set consistent y-axis limits
            if i == 0:  # Only show legend on first plot
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('individual_action_f1_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_action_auc_pr_plots(log_df):
    """Create separate AUC-PR plots for each action"""
    actions = sorted(log_df['action'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Action AUC-PR Performance by Model Type', fontsize=16, fontweight='bold')
    
    colors = {
        'unweighted': '#1f77b4',
        'unweighted_w_abstraction': '#ff7f0e', 
        'weighted_1object': '#2ca02c'
    }
    
    # Plot AUC-PR for each action
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
            ax.set_title(f'{action.upper()} - AUC-PR')
            ax.set_ylim(0, 1)  # Set consistent y-axis limits
            if i == 0:  # Only show legend on first plot
                ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('individual_action_auc_pr_performance.png', dpi=300, bbox_inches='tight')
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
        
        print("\nCreating individual action F1 plots...")
        create_individual_action_f1_plots(log_df)
        
        print("\nCreating individual action AUC-PR plots...")
        create_individual_action_auc_pr_plots(log_df)
        
        print_summary_statistics(eval_df, log_df)
        print("\nVisualizations saved as:")
        print("  - rdn_model_comparison.png (main comparison: weighted F1 + average AUC-PR)")
        print("  - individual_action_f1_performance.png (individual action F1 scores)")
        print("  - individual_action_auc_pr_performance.png (individual action AUC-PR scores)")
    else:
        print("No data found to visualize!")
