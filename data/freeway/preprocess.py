"""
Preprocess Freeway relationship data for learning.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from srlearn import Database, Background
import string
import os
import shutil

# Freeway-specific modes (relationship predicates)
modes = [
    "carAbove(+state, -car).",
    "carBelow(+state, -car).",
    "carDirectlyAbove(+state, +car).",
    "carDirectlyBelow(+state, +car).",
    "nearbyCar(+state, +car).",
    "sameLevelAsCar(+state, -car).",
    "leftOfCar(+state, +car).",
    "rightOfCar(+state, +car).",
    "carFacingSide(+state, +car, #direction).",
    "action(+state, #name)."
]

# Bridgers for Freeway (if needed)
bridgers = []

# Freeway action space (3 actions: NOOP=0, UP=1, DOWN=2)
primitive_actions = ["noop", "up", "down"]

# Lowercase everything for consistency
bridgers = [bridger.lower() for bridger in bridgers]
modes = [mode.lower() for mode in modes]

# Argument parser
parser = argparse.ArgumentParser(description="Process Freeway relationship file")
parser.add_argument("--file", type=str, default="", help="Relationship file (combined_freeway_data.txt)")
parser.add_argument("--node_size", type=int, default=2, help="Node size for background")
parser.add_argument("--max_tree_depth", type=int, default=3, help="Max tree depth for background")
parser.add_argument("--remove_0_weights", action="store_true", help="Remove facts with 0.00 weights")
parser.add_argument("--all", type=bool, default=False, help="Process all trajectories together")
args = parser.parse_args()

# Set base directory
if args.all:
    base_dir = "data/freeway/all"
else:
    file = args.file.split("/")[-1]
    file_parts = file.split("_")
    # Extract trajectory ID from filename
    traj_id = "_".join(file_parts[:3]) if len(file_parts) >= 3 else file_parts[0]
    base_dir = f"data/freeway/single_t/{traj_id}"

print(f"Processing: {args.file}")
print(f"Output directory: {base_dir}")

# Create directories for each action
for action in primitive_actions:
    action_dir = f"{base_dir}/{action}"
    if os.path.exists(action_dir):
        shutil.rmtree(action_dir)
    os.makedirs(f"{action_dir}/train", exist_ok=True)
    os.makedirs(f"{action_dir}/test", exist_ok=True)

# Load data
if args.all:
    df = pd.read_csv(args.file, delimiter="\t")
else:
    df = pd.read_csv(args.file, delimiter="\t")

# Filter to valid Freeway actions only (0, 1, 2)
df = df[df['action'] <= 2]

print(f"Total frames: {len(df)}")
print(f"Action distribution:\n{df['action'].value_counts()}")

# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv(f"{base_dir}/train.csv", index=False)
test.to_csv(f"{base_dir}/test.csv", index=False)

print(f"Train samples: {len(train)}")
print(f"Test samples: {len(test)}")

# Freeway action mapping
actions = {
    0: "noop",
    1: "down",
    2: "up"
}

# Initialize data structures
train_action_files = {action: [[], []] for action in primitive_actions}
test_action_files = {action: [[], []] for action in primitive_actions}

# Process training data
print("\nProcessing training data...")
for _, row in train.iterrows():
    # Create state ID from global_frame_id or qframe_id
    if 'global_frame_id' in row:
        s_id = f"s{row['global_frame_id']}"
    else:
        s_id = "s" + str(row["qframe_id"]).replace("_", "")
    
    action_code = int(row['action'])
    action_name = actions.get(action_code)
    
    if action_name is None:
        continue
    
    # For each primitive action, create positive/negative examples
    for pa in primitive_actions:
        action_str = f"action({s_id}, {pa})."
        action_str = action_str.lower()
        
        if pa == action_name:
            # Positive example: this action was taken
            train_action_files[pa][0].append(action_str)
        else:
            # Negative example: add OTHER actions that were NOT taken
            for other_action in primitive_actions:
                if other_action != pa:
                    other_action_str = f"action({s_id}, {other_action})."
                    other_action_str = other_action_str.lower()
                    train_action_files[pa][1].append(other_action_str)

# Write train action files
for pa in primitive_actions:
    with open(f"{base_dir}/{pa}/train/train_pos.txt", "w") as f:
        f.write("\n".join(train_action_files[pa][0]))
    with open(f"{base_dir}/{pa}/train/train_neg.txt", "w") as f:
        f.write("\n".join(train_action_files[pa][1]))

print("Training action files written.")

# Process test data
print("\nProcessing test data...")
for _, row in test.iterrows():
    if 'global_frame_id' in row:
        s_id = f"s{row['global_frame_id']}"
    else:
        s_id = "s" + str(row["qframe_id"]).replace("_", "")
    
    action_code = int(row['action'])
    action_name = actions.get(action_code)
    
    if action_name is None:
        continue
    
    for pa in primitive_actions:
        action_str = f"action({s_id}, {pa})."
        action_str = action_str.lower()
        
        if pa == action_name:
            test_action_files[pa][0].append(action_str)
        else:
            test_action_files[pa][1].append(action_str)

# Write test action files
for pa in primitive_actions:
    with open(f"{base_dir}/{pa}/test/test_pos.txt", "w") as f:
        f.write("\n".join(test_action_files[pa][0]))
    with open(f"{base_dir}/{pa}/test/test_neg.txt", "w") as f:
        f.write("\n".join(test_action_files[pa][1]))

print("Test action files written.")

# Process facts (relationships)
print("\nProcessing relationship facts...")
for pa in primitive_actions:
    # Train facts
    with open(f"{base_dir}/{pa}/train/train_facts.txt", 'w') as f:
        with open(f"{base_dir}/{pa}/train/fact_weights.txt", 'w') as f2:
            for _, row in train.iterrows():
                if 'global_frame_id' in row:
                    s_id = f"s{row['global_frame_id']}"
                else:
                    s_id = "s" + str(row["qframe_id"]).replace("_", "")
                
                rels = str(row["relationships"])
                weights = str(row.get("predicate_weights", ""))
                
                weights_list = []
                if weights != "nan" and weights:
                    weights_list = weights.split(" ")
                
                if rels != "nan" and rels:
                    rels_list = rels.split(" , ")
                    
                    for i, rel in enumerate(rels_list):
                        rel = rel.strip()
                        if rel:
                            # Add state as first argument
                            if "(" not in rel:
                                rel = rel + "()"
                            rel = rel.replace("(", f"({s_id},")
                            rel = rel.replace(",)", ")")
                            
                            if not rel.endswith("."):
                                rel += "."
                            
                            rel = rel.lower()
                            rel = rel.replace("_", "")
                            
                            f.write(rel + "\n")
                            
                            # Write weight if available
                            if i < len(weights_list):
                                f2.write(rel + " " + weights_list[i] + "\n")
    
    # Test facts
    with open(f"{base_dir}/{pa}/test/test_facts.txt", 'w') as f:
        for _, row in test.iterrows():
            if 'global_frame_id' in row:
                s_id = f"s{row['global_frame_id']}"
            else:
                s_id = "s" + str(row["qframe_id"]).replace("_", "")
            
            rels = str(row["relationships"])
            
            if rels != "nan" and rels:
                for rel in rels.split(" , "):
                    rel = rel.strip()
                    if rel:
                        if "(" not in rel:
                            rel = rel + "()"
                        rel = rel.replace("(", f"({s_id},")
                        rel = rel.replace(",)", ")")
                        
                        if not rel.endswith("."):
                            rel += "."
                        
                        rel = rel.lower()
                        rel = rel.replace("_", "")
                        
                        f.write(rel + "\n")

print("Relationship facts written.")

# Function to remove facts with 0 weights from train_facts.txt only
def remove_zero_weight_facts(train_facts_file, weights_file):
    """Remove facts with 0.00 weights from the train_facts file using the weights file"""
    if not os.path.exists(weights_file):
        print(f"Warning: {weights_file} not found. Skipping zero weight removal.")
        return
    
    zero_weight_facts = set()
    with open(weights_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ' ' in line:
                parts = line.split(' ')
                if len(parts) >= 2:
                    fact = parts[0]
                    weight = parts[1]
                    if weight == '0.00':
                        zero_weight_facts.add(fact)
    
    with open(train_facts_file, 'r') as f:
        facts = f.read().splitlines()
    
    filtered_facts = []
    removed_count = 0
    for fact in facts:
        fact_without_dot = fact.rstrip('.')
        if fact_without_dot not in zero_weight_facts:
            filtered_facts.append(fact)
        else:
            removed_count += 1
    
    with open(train_facts_file, 'w') as f:
        f.write('\n'.join(filtered_facts))
    
    print(f"Removed {removed_count} facts with 0.00 weights from {train_facts_file}")

# Remove zero weight facts if requested
if args.remove_0_weights:
    print("\nRemoving facts with 0.00 weights from train_facts.txt files...")
    for pa in primitive_actions:
        train_facts_file = f"{base_dir}/{pa}/train/train_facts.txt"
        weights_file = f"{base_dir}/{pa}/train/fact_weights.txt"
        remove_zero_weight_facts(train_facts_file, weights_file)

# Create background knowledge files
print("\nCreating background knowledge files...")
for action in primitive_actions:
    bk = Background(modes=modes, bridgers=bridgers, number_of_clauses=20, number_of_cycles=20)
    
    with open(f"{base_dir}/{action}/train/train_bk.txt", "w") as f:
        f.write(str(bk))
    
    with open(f"{base_dir}/{action}/test/test_bk.txt", "w") as f:
        f.write(str(bk))

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"Output directory: {base_dir}")
print(f"Actions processed: {', '.join(primitive_actions)}")
print(f"Train samples: {len(train)}")
print(f"Test samples: {len(test)}")
print("="*60)
