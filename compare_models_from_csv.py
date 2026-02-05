import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)
    return pd.read_csv(csv_path)

def filter_data(df, filters, action=None):
    filtered_df = df.copy()
    
    # Apply CLI filters
    if filters:
        filter_pairs = filters.split(',')
        for pair in filter_pairs:
            if '=' not in pair:
                print(f"Warning: Invalid filter format '{pair}'. Ignored.")
                continue
                
            key, value_str = pair.split('=', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            # Support OR condition with |
            target_values = value_str.split('|')
            processed_values = []
            
            for v in target_values:
                # Try to convert value to appropriate type
                if v == "N/A":
                    processed_values.append(float('nan')) # Handle pandas NaN
                    processed_values.append(v) # Also keep string just in case
                    continue
                    
                try:
                    if '.' in v:
                        processed_values.append(float(v))
                    else:
                        processed_values.append(int(v))
                except ValueError:
                    processed_values.append(v) # Keep as string
            
            if key in filtered_df.columns:
                # Use isin for multiple values
                filtered_df = filtered_df[filtered_df[key].isin(processed_values)]
            else:
                print(f"Warning: Column '{key}' not found in CSV. Skipping filter.")
                
    # Apply Action filter
    if action:
        if "Action" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Action"] == action]
        else:
            print("Warning: 'Action' column not found. Skipping action filter.")
            
    return filtered_df

def plot_data(df, metric, groupby_cols, output_file, title_suffix=""):
    plt.figure(figsize=(10, 6))
    
    # Determine x-axis and hue
    x_col = groupby_cols[0]
    hue_col = groupby_cols[1] if len(groupby_cols) > 1 else None
    
    # Check if cols exist
    if x_col not in df.columns:
        print(f"Error: Column '{x_col}' not found for grouping.")
        return
        
    if hue_col and hue_col not in df.columns:
        print(f"Warning: Hue column '{hue_col}' not found. ignoring.")
        hue_col = None

    try:
        # Seaborn Barplot with errorbar="sd" (Standard Deviation)
        sns.barplot(data=df, x=x_col, y=metric, hue=hue_col, errorbar="sd", capsize=.1)
        
        plt.title(f"Comparison of {metric} {title_suffix}")
        plt.xlabel(x_col)
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare models from CSV results.")
    parser.add_argument("--csv", default="experiment_results.csv", help="Path to results CSV")
    parser.add_argument("--metric", default="AUC-PR", help="Metric to plot (e.g., F1, AUC-PR)")
    parser.add_argument("--groupby", default="Action,Model", help="Comma-separated columns to group by (x-axis, hue). Default: Action,Model")
    parser.add_argument("--filters", default="Train Neg-to-Pos=2.0,Test Neg-to-Pos=2.0", help="Comma-separated key=value pairs (e.g., 'Parameters (Num Trees)=1')")
    parser.add_argument("--action", help="Filter by Action")
    parser.add_argument("--per_action", action="store_true", help="Generate separate plots for each action")
    parser.add_argument("--output", default="comparison_plot.png", help="Output filename")
    
    args = parser.parse_args()
    
    df = load_data(args.csv)
    
    filtered_df = filter_data(df, args.filters, args.action)
    
    if filtered_df.empty:
        print("No data matches the specified filters.")
        return

    # Calculate run counts per model and update Model column
    if "Model" in filtered_df.columns and "Seed" in filtered_df.columns:
        seed_counts = filtered_df.groupby("Model")["Seed"].nunique()
        
        def append_info(row):
            model = row["Model"]
            count = seed_counts.get(model, 0)
            label = f"{model}"
            
            # Add Lambda if present
            if "Lambda" in row and pd.notna(row["Lambda"]) and row["Lambda"] != 'N/A':
                 label += f" (λ={row['Lambda']})"
            
            # Add Grounding Penalty if present (parsing from experiment name as it might be N/A in params if not consistently parsed)
            # Or use the parsed value if we add it to CSV. 
            # In aggregate_results.py we parsed "grounding_penalty" but didn't add it to CSV explicit column?
            # Let's check CSV headers. It wasn't in headers.
            # But we can parse it from 'Experiment' column here as a fallback or update aggregation.
            # Updating aggregation is cleaner but let's check if we can get it from Experiment.
            experiment = row.get("Experiment", "")
            if "grounding_penalty" in experiment:
                try:
                    parts = experiment.split("_")
                    idx = parts.index("penalty")
                    penalty = parts[idx+1]
                    label += f" (gp={penalty})"
                except:
                    pass

            label += f" [{count} runs]"
            return label
            
        filtered_df["Model"] = filtered_df.apply(append_info, axis=1)

    print(f"Comparing {len(filtered_df)} runs...")
    
    groupby_cols = args.groupby.split(',')
    title_suffix = f"({args.filters})" if args.filters else ""
    if args.action:
        title_suffix += f" Action: {args.action}"

    if args.per_action:
        if "Action" not in filtered_df.columns:
            print("Error: 'Action' column not found, cannot split by action.")
            return

        actions = filtered_df["Action"].unique()
        print(f"Generating plots for actions: {actions}")
        
        base_name, ext = os.path.splitext(args.output)
        
        for action in actions:
            action_df = filtered_df[filtered_df["Action"] == action]
            if action_df.empty:
                continue
                
            out_file = f"{base_name}_{action}{ext}"
            action_suffix = f"{title_suffix} Action: {action}"
            print(f"Plotting for action: {action} -> {out_file}")
            plot_data(action_df, args.metric, groupby_cols, out_file, action_suffix)
            
    else:     
        plot_data(filtered_df, args.metric, groupby_cols, args.output, title_suffix)

if __name__ == "__main__":
    main()
