#!/usr/bin/env python3
"""
Visualization script for RDN model results.
Compares different model configurations based on depth, trees, and weighting strategies.
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_infer_log(log_path: str) -> Dict:
    """
    Parse inference log file to extract F1 score and other metrics.
    
    Returns dict with keys: f1, precision, recall, auc_roc, auc_pr
    """
    metrics = {}
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Extract F1 score
        f1_match = re.search(r'%\s+F1\s+=\s+([\d.]+)', content)
        if f1_match:
            metrics['f1'] = float(f1_match.group(1))
            
        # Extract Precision
        prec_match = re.search(r'%\s+Precision\s+=\s+([\d.]+)', content)
        if prec_match:
            metrics['precision'] = float(prec_match.group(1))
            
        # Extract Recall
        recall_match = re.search(r'%\s+Recall\s+=\s+([\d.]+)', content)
        if recall_match:
            metrics['recall'] = float(recall_match.group(1))
            
        # Extract AUC ROC
        auc_roc_match = re.search(r'%\s+AUC ROC\s+=\s+([\d.]+)', content)
        if auc_roc_match:
            metrics['auc_roc'] = float(auc_roc_match.group(1))
            
        # Extract AUC PR
        auc_pr_match = re.search(r'%\s+AUC PR\s+=\s+([\d.]+)', content)
        if auc_pr_match:
            metrics['auc_pr'] = float(auc_pr_match.group(1))
            
    except FileNotFoundError:
        print(f"Warning: File not found: {log_path}")
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        
    return metrics


def parse_eval_report(report_path: str) -> Dict:
    """
    Parse eval_report.txt to extract weighted F1 scores for train and test.
    
    Returns dict with keys: train_weighted_f1, test_weighted_f1
    """
    metrics = {}
    
    try:
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Extract training weighted F1
        train_match = re.search(r'TRAINING SET PERFORMANCE.*?weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
        if train_match:
            metrics['train_weighted_f1'] = float(train_match.group(3))
            
        # Extract test weighted F1
        test_match = re.search(r'TEST SET PERFORMANCE.*?METHOD 1.*?weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
        if test_match:
            metrics['test_weighted_f1'] = float(test_match.group(3))
            
    except FileNotFoundError:
        print(f"Warning: File not found: {report_path}")
    except Exception as e:
        print(f"Error parsing {report_path}: {e}")
        
    return metrics


def parse_folder_name(folder_name: str) -> Dict:
    """
    Extract configuration details from folder name.
    
    Returns dict with keys: trees, depth, weighting_type, trajectory_type
    """
    config = {
        'trees': None,
        'depth': None,
        'weighting_type': 'none',
        'trajectory_type': 'single'
    }
    
    # Extract number of trees
    trees_match = re.search(r'trees_(\d+)', folder_name)
    if trees_match:
        config['trees'] = int(trees_match.group(1))
        
    # Extract depth
    depth_match = re.search(r'depth_(\d+)', folder_name)
    if depth_match:
        config['depth'] = int(depth_match.group(1))
        
    # Determine weighting type
    if 'per_example_weight' in folder_name:
        config['weighting_type'] = 'per_example_weight'
    elif 'weighted_1object' in folder_name:
        config['weighting_type'] = 'weighted_1object'
    elif 'w_abstraction' in folder_name:
        config['weighting_type'] = 'w_abstraction'
    else:
        config['weighting_type'] = 'none'
        
    # Determine trajectory type
    if '_all' in folder_name or folder_name.endswith('_all'):
        config['trajectory_type'] = 'all'
    else:
        config['trajectory_type'] = 'single'
        
    return config


def collect_results(base_path: str, trajectory_filter: str = None, 
                    trees_filter: int = None, depth_filter: int = None) -> Dict:
    """
    Collect all results from RDN model directories.
    
    Args:
        base_path: Base directory containing model results
        trajectory_filter: 'all', 'single', or None for both
        trees_filter: Filter by number of trees, or None for all
        depth_filter: Filter by depth, or None for all
        
    Returns:
        Dictionary with results organized by configuration
    """
    results = defaultdict(lambda: defaultdict(dict))
    actions = ['noop', 'fire', 'up', 'right', 'left', 'down']
    
    base_path = Path(base_path)
    
    # Iterate through all model directories
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        folder_name = model_dir.name
        
        # Handle single_trajectory subdirectory
        if folder_name == 'single_trajectory':
            for sub_model_dir in model_dir.iterdir():
                if not sub_model_dir.is_dir():
                    continue
                process_model_directory(sub_model_dir, results, actions, 
                                      trajectory_filter, trees_filter, depth_filter)
        else:
            process_model_directory(model_dir, results, actions, 
                                  trajectory_filter, trees_filter, depth_filter)
    
    return results


def process_model_directory(model_dir: Path, results: Dict, actions: List[str],
                           trajectory_filter: str, trees_filter: int, depth_filter: int):
    """Process a single model directory and extract metrics."""
    folder_name = model_dir.name
    config = parse_folder_name(folder_name)
    
    # Apply filters
    if trajectory_filter and config['trajectory_type'] != trajectory_filter:
        return
    if trees_filter and config['trees'] != trees_filter:
        return
    if depth_filter and config['depth'] != depth_filter:
        return
        
    # Skip if essential config is missing
    if config['trees'] is None or config['depth'] is None:
        return
    
    # Create config key
    config_key = (config['trees'], config['depth'], 
                  config['weighting_type'], config['trajectory_type'])
    
    # Parse eval_report.txt for overall metrics
    eval_report_path = model_dir / 'eval_report.txt'
    if eval_report_path.exists():
        eval_metrics = parse_eval_report(str(eval_report_path))
        results[config_key]['overall'] = eval_metrics
    
    # Parse individual action inference logs
    for action in actions:
        action_dir = model_dir / action
        infer_log = action_dir / f"{action}_infer.log"
        
        if infer_log.exists():
            action_metrics = parse_infer_log(str(infer_log))
            results[config_key][action] = action_metrics


def plot_depth_vs_f1(results: Dict, output_path: str = 'depth_vs_f1.png'):
    """
    Plot depth on x-axis vs weighted F1 score on y-axis for train and test on same plot.
    Train uses dotted lines, test uses solid lines.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Organize data by weighting type and trajectory type
    data_by_config = defaultdict(lambda: defaultdict(lambda: {'depths': [], 'train_f1': [], 'test_f1': []}))
    
    for config_key, metrics in results.items():
        trees, depth, weighting, trajectory = config_key
        
        if 'overall' in metrics:
            train_f1 = metrics['overall'].get('train_weighted_f1')
            test_f1 = metrics['overall'].get('test_weighted_f1')
            
            if train_f1 is not None and test_f1 is not None:
                label = f"{weighting}_{trajectory}"
                data_by_config[label][trees]['depths'].append(depth)
                data_by_config[label][trees]['train_f1'].append(train_f1)
                data_by_config[label][trees]['test_f1'].append(test_f1)
    
    # Plot both train and test on same plot
    for label, trees_data in sorted(data_by_config.items()):
        for trees, data in sorted(trees_data.items()):
            # Sort by depth
            sorted_indices = np.argsort(data['depths'])
            depths = np.array(data['depths'])[sorted_indices]
            train_f1 = np.array(data['train_f1'])[sorted_indices]
            test_f1 = np.array(data['test_f1'])[sorted_indices]
            
            config_label = f"{label} ({trees} trees)"
            
            # Plot training with dotted line
            ax.plot(depths, train_f1, marker='o', linestyle='--', label=f"{config_label} - Train", linewidth=2, alpha=0.7)
            
            # Plot test with solid line
            ax.plot(depths, test_f1, marker='s', linestyle='-', label=f"{config_label} - Test", linewidth=2)
    
    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Weighted F1 Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Depth vs Weighted F1 Score (Train and Test)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    
    # Set x-ticks but don't limit axes
    all_depths = sorted(set(d for td in data_by_config.values() for t in td.values() for d in t['depths']))
    if all_depths:
        ax.set_xticks(all_depths)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved depth vs F1 plot to {output_path}")
    plt.close()


def plot_action_f1_comparison(results: Dict, output_path: str = 'action_f1_comparison.png'):
    """
    Compare F1 scores for each action across different configurations.
    """
    actions = ['noop', 'fire', 'up', 'right', 'left', 'down']
    
    # Organize data by configuration
    data_by_config = defaultdict(lambda: {action: [] for action in actions})
    config_labels = []
    
    for config_key, metrics in sorted(results.items()):
        trees, depth, weighting, trajectory = config_key
        label = f"T{trees}_D{depth}_{weighting[:3]}_{trajectory[:3]}"
        config_labels.append(label)
        
        for action in actions:
            if action in metrics:
                f1 = metrics[action].get('f1', 0)
                data_by_config[label][action].append(f1)
            else:
                data_by_config[label][action].append(0)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(config_labels))
    width = 0.13
    
    for i, action in enumerate(actions):
        values = [data_by_config[label][action][0] if data_by_config[label][action] else 0 
                 for label in config_labels]
        offset = width * (i - len(actions) / 2)
        ax.bar(x + offset, values, width, label=action)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Action F1 Score Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved action F1 comparison plot to {output_path}")
    plt.close()


def plot_action_f1_by_depth(results: Dict, output_path: str = 'action_f1_by_depth.png'):
    """
    Create line charts of F1 scores for each action by depth.
    """
    actions = ['noop', 'fire', 'up', 'right', 'left', 'down']
    
    # Group by trajectory type and weighting
    for trajectory in ['all', 'single']:
        for weighting in ['none', 'per_example_weight', 'weighted_1object', 'w_abstraction']:
            # Collect data for this configuration
            depth_action_f1 = defaultdict(lambda: {action: [] for action in actions})
            
            for config_key, metrics in results.items():
                trees, depth, weight_type, traj_type = config_key
                
                if traj_type != trajectory or weight_type != weighting:
                    continue
                
                for action in actions:
                    if action in metrics:
                        f1 = metrics[action].get('f1', 0)
                        depth_action_f1[depth][action].append(f1)
            
            if not depth_action_f1:
                continue
            
            # Create line chart
            depths = sorted(depth_action_f1.keys())
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for action in actions:
                f1_scores = []
                valid_depths = []
                for depth in depths:
                    if depth_action_f1[depth][action]:
                        f1_scores.append(np.mean(depth_action_f1[depth][action]))
                        valid_depths.append(depth)
                
                if valid_depths:
                    ax.plot(valid_depths, f1_scores, marker='o', label=action, linewidth=2)
            
            ax.set_xlabel('Depth', fontsize=12)
            ax.set_ylabel('F1 Score', fontsize=12)
            ax.set_title(f'Action F1 Scores by Depth: {trajectory} trajectories, {weighting} weighting', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if depths:
                ax.set_xticks(depths)
            
            plt.tight_layout()
            filename = output_path.replace('.png', f'_{trajectory}_{weighting}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved action F1 line chart to {filename}")
            plt.close()


def print_summary_table(results: Dict):
    """Print a summary table of all results."""
    print("\n" + "="*120)
    print("RDN MODEL RESULTS SUMMARY")
    print("="*120)
    print(f"{'Trees':<6} {'Depth':<6} {'Weighting':<20} {'Trajectory':<12} {'Train F1':<10} {'Test F1':<10}")
    print("-"*120)
    
    for config_key, metrics in sorted(results.items()):
        trees, depth, weighting, trajectory = config_key
        
        if 'overall' in metrics:
            train_f1 = metrics['overall'].get('train_weighted_f1', 0)
            test_f1 = metrics['overall'].get('test_weighted_f1', 0)
            
            print(f"{trees:<6} {depth:<6} {weighting:<20} {trajectory:<12} {train_f1:<10.4f} {test_f1:<10.4f}")
    
    print("="*120)
    
    # Action-specific summary
    print("\nACTION-SPECIFIC F1 SCORES")
    print("="*120)
    
    actions = ['noop', 'fire', 'up', 'right', 'left', 'down']
    header = f"{'Configuration':<40}"
    for action in actions:
        header += f" {action:<8}"
    print(header)
    print("-"*120)
    
    for config_key, metrics in sorted(results.items()):
        trees, depth, weighting, trajectory = config_key
        config_str = f"T{trees}_D{depth}_{weighting[:10]}_{trajectory}"
        
        row = f"{config_str:<40}"
        for action in actions:
            if action in metrics:
                f1 = metrics[action].get('f1', 0)
                row += f" {f1:<8.4f}"
            else:
                row += f" {'-':<8}"
        print(row)
    
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description='Visualize RDN model results')
    parser.add_argument('--base_path', type=str, 
                       default='/home/nikhilesh/Projects/NeSY-Imitation-Learning/rdn_models/seaquest',
                       help='Base path to RDN model results')
    parser.add_argument('--trajectory', type=str, choices=['all', 'single', 'both'], 
                       default='both',
                       help='Filter by trajectory type')
    parser.add_argument('--trees', type=int, default=None,
                       help='Filter by number of trees')
    parser.add_argument('--depth', type=int, default=None,
                       help='Filter by depth')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Determine trajectory filter
    trajectory_filter = None if args.trajectory == 'both' else args.trajectory
    
    # Collect results
    print(f"Collecting results from {args.base_path}...")
    results = collect_results(args.base_path, trajectory_filter, 
                             args.trees, args.depth)
    
    if not results:
        print("No results found matching the specified filters.")
        return
    
    print(f"Found {len(results)} configurations")
    
    # Print summary
    print_summary_table(results)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    plot_depth_vs_f1(results, str(output_dir / 'depth_vs_f1.png'))
    plot_action_f1_comparison(results, str(output_dir / 'action_f1_comparison.png'))
    plot_action_f1_by_depth(results, str(output_dir / 'action_f1_by_depth.png'))
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
