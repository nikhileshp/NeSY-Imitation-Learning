# RDN Model Results Visualization

This document explains how to use the `visualize_rdn_results.py` script to visualize and compare RDN model results.

## Overview

The script parses RDN model results from various configurations and creates comprehensive visualizations comparing:
- Different tree depths (1-10)
- Number of trees (1, 5, 10)
- Weighting strategies (none, per_example_weight, weighted_1object)
- Trajectory types (all trajectories vs single trajectory)

## Usage

### Basic Usage

```bash
# Visualize all results (default)
python visualize_rdn_results.py

# Visualize only "all trajectories" results
python visualize_rdn_results.py --trajectory all

# Visualize only "single trajectory" results
python visualize_rdn_results.py --trajectory single

# Visualize both with comparison
python visualize_rdn_results.py --trajectory both
```

### Advanced Filtering

```bash
# Filter by number of trees
python visualize_rdn_results.py --trees 1

# Filter by depth
python visualize_rdn_results.py --depth 5

# Combine filters
python visualize_rdn_results.py --trajectory all --trees 1 --depth 4

# Specify custom output directory
python visualize_rdn_results.py --output_dir results_visualizations
```

### Command-Line Arguments

- `--base_path`: Base path to RDN model results (default: `rdn_models/seaquest`)
- `--trajectory`: Filter by trajectory type (`all`, `single`, or `both`)
- `--trees`: Filter by number of trees (e.g., `1`, `5`, `10`)
- `--depth`: Filter by depth (e.g., `1`, `2`, `3`, etc.)
- `--output_dir`: Output directory for plots (default: current directory)

## Output Files

The script generates three types of visualizations:

### 1. Depth vs F1 Score (`depth_vs_f1.png`)
- Two side-by-side plots showing training and test performance
- X-axis: Model depth
- Y-axis: Weighted F1 score
- Multiple lines for different weighting strategies and trajectory types

### 2. Action F1 Comparison (`action_f1_comparison.png`)
- Grouped bar chart comparing F1 scores for each action across configurations
- Shows performance for: noop, fire, up, right, left, down
- Each configuration is labeled with trees, depth, weighting type, and trajectory type

### 3. Action F1 Heatmaps (`action_f1_heatmap_*.png`)
- Separate heatmaps for each combination of trajectory type and weighting strategy
- Rows: Actions (noop, fire, up, right, left, down)
- Columns: Depth values
- Color intensity indicates F1 score (darker = better)

## Data Sources

The script parses two types of files:

### 1. `eval_report.txt`
Located in each model configuration directory (e.g., `negpos_2_trees_1_depth_4_all/eval_report.txt`)
- Extracts: Train weighted F1, Test weighted F1

### 2. `{action}_infer.log`
Located in action subdirectories (e.g., `negpos_2_trees_1_depth_4_all/noop/noop_infer.log`)
- Extracts: F1, Precision, Recall, AUC ROC, AUC PR per action

## Folder Name Conventions

The script automatically parses folder names to extract configuration details:

- **Trees**: `trees_X` → Number of trees (e.g., `trees_1`, `trees_5`)
- **Depth**: `depth_X` → Tree depth (e.g., `depth_4`, `depth_8`)
- **Weighting**: 
  - `per_example_weight` → Example-based weighting
  - `weighted_1object` → 1.0 weight for nearest object, 0 for others
  - `w_abstraction` → With abstraction
  - No keyword → No weighting
- **Trajectory**:
  - `_all` → Trained on all trajectories
  - In `single_trajectory/` folder → Trained on single trajectory

## Example Workflows

### Compare All Trajectories vs Single Trajectory

```bash
# Create separate visualizations
python visualize_rdn_results.py --trajectory all --output_dir all_traj_results
python visualize_rdn_results.py --trajectory single --output_dir single_traj_results

# Or create combined comparison
python visualize_rdn_results.py --trajectory both --output_dir comparison_results
```

### Analyze Specific Configuration

```bash
# Analyze 1-tree models only
python visualize_rdn_results.py --trees 1 --output_dir one_tree_analysis

# Analyze depth=5 models only
python visualize_rdn_results.py --depth 5 --output_dir depth5_analysis

# Analyze single trajectory with 1 tree
python visualize_rdn_results.py --trajectory single --trees 1 --output_dir single_1tree
```

## Summary Tables

The script also prints detailed summary tables to the console:

1. **Overall Performance Table**: Shows train/test weighted F1 for each configuration
2. **Action-Specific Table**: Shows F1 scores for each action across all configurations

## Key Findings

Based on the visualizations, you can identify:

1. **Optimal depth**: Which depth provides best performance for each configuration
2. **Weighting impact**: How different weighting strategies affect performance
3. **Trajectory comparison**: Whether training on all trajectories or single trajectory performs better
4. **Action-specific insights**: Which actions are easier/harder to learn
5. **Overfitting**: Compare train vs test performance to detect overfitting

## Notes

- Missing values in the output indicate that the corresponding log file was not found
- Zero values may indicate missing eval_report.txt files
- The script is robust to missing files and will continue processing available data
- All plots are saved at 300 DPI for publication quality

## Troubleshooting

### No results found
- Check that `--base_path` points to the correct directory
- Verify that model directories follow the expected naming convention

### Missing metrics
- Ensure `eval_report.txt` exists in model directories
- Ensure `{action}_infer.log` files exist in action subdirectories

### Empty plots
- Check that filters (`--trajectory`, `--trees`, `--depth`) aren't too restrictive
- Verify that model training and inference completed successfully
