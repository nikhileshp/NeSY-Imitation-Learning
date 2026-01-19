import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from srlearn import Database, Background
import string
import os

modes=["aboveOfDiver(+state, +diver).",
"aboveOfEnemy(+state, +enemy).",
"aboveOfMissile(+state, +missile). ",
"aboveOfSubmarine(+state, +submarine).",
"aboveWater_surface(+state).",
"belowOfDiver(+state, +diver).",
"belowOfEnemy(+state, +enemy).",
"belowOfMissile(+state, +missile).",
"belowOfSubmarine(+state, +submarine).",
"belowWater_surface(+state).",
"diversEmpty(+state).",
"diversNotfull(+state).",
"diversfull(+state).",
"enemyFacingLeft(+state, +submarine).",
"enemyFacingRight(+state, +submarine).",
"facingLeft(+state).",
"facingRight(+state).",
"leftOfDiver(+state, +diver).",
"leftOfEnemy(+state, +enemy).",
"leftOfMissile(+state, +missile).",
"leftOfSubmarine(+state, +submarine).",
"nearbyDiver(+state, +diver).",
"nearbyEnemy(+state, +enemy).",
"nearbyMissile(+state, +missile).",
"nearbySubmarine(+state, +submarine).",
"oxygenOk(+state).",
"rightOfDiver(+state, +diver).",
"rightOfEnemy(+state, +enemy).",
"rightOfMissile(+state, +missile).",
"rightOfSubmarine(+state, +submarine).",
"sameLevelAsDiver(+state, +diver).",
"sameLevelAsEnemy(+state, +enemy).",
"sameLevelAsMissile(+state, +missile).",
"sameLevelAsSubmarine(+state, +submarine).",
"visibleDiver(+state, -diver).",
"visibleEnemy(+state, -enemy).",
"visibleEnemySubmarine(+state, -submarine).",
"visibleMissile(+state, -missile).",
"action(+state, #name)."
]

privileged_modes = ["inradiusEnemy(+state, -enemy)",
                    "inradiusMissile(+state, -missile)",
                    "inradiusSubmarine(+state, -submarine)",
                    "inradiusDiver(+state, -diver)"]

priviledged_modes_removed = ["visibleEnemy(+state, -enemy)",
                            "visibleMissile(+state, -missile)",
                            "visibleSubmarine(+state, -submarine)",
                            "visibleDiver(+state, -diver)"]


all_privileged_modes = modes + privileged_modes 
#remove priviledged modes_removed from all_privileged_modes
all_privileged_modes = [mode for mode in all_privileged_modes if mode not in priviledged_modes_removed]


privileged_bridgers = ["inRadiusEnemy/2",
                    "inRadiusMissile/2",
                    "inRadiusSubmarine/2",
                    "inRadiusDiver/2"]

bridgers = ["vissibleMissile/2",
"vissibleEnemy/2",
"vissibleEnemySubmarine/2",
"vissibleDiver/2"]

primitive_actions = ["noop","fire","up","right","left","down"]

bridgers = [bridger.lower() for bridger in bridgers]
modes = [mode.lower() for mode in modes]

THRESHOLD = 0.70


parser = argparse.ArgumentParser(description="Process relationship file")
parser.add_argument("--file", type=str, default="", help="Relationship file")
parser.add_argument("--node_size", type=str, default=2, help="Node size for background")
parser.add_argument("--max_tree_depth", type=str, default=3, help="Max tree depth for background")
parser.add_argument("--remove_0_weights", action="store_true", help="Remove facts with 0.00 weights from train_facts.txt using fact_weights.tsv")
parser.add_argument("--all", type=bool, default=False, help="Process all actions together")


args = parser.parse_args()
if args.all:
    base_dir = "data/seaquest/all"
else:
    file = args.file.split("/")[-1]
    file = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2]
    base_dir = "data/seaquest/single_t/"+file
print(args.file)



for action in primitive_actions:
  if os.path.exists(f"{base_dir}/{action}"):
    import shutil
    shutil.rmtree(f"{base_dir}/{action}")
  os.makedirs(f"{base_dir}/{action}/train", exist_ok=True)
  os.makedirs(f"{base_dir}/{action}/test", exist_ok=True)
# df = pd.read_csv(args.file)
if args.all:
  df = pd.read_csv(args.file, delimiter="\t")
else:
  df = pd.read_csv(args.file)
df = df[df['action'] <= 5]

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

actions = {
    0: "noop",
    1: "fire",
    2: "up",
    3: "right",
    4: "left",
    5: "down",
    6: ("up", "right"),
    7: ("up", "left"),
    8: ("down", "right"),
    9: ("down", "left"),
    10: ("up", "fire"),
    11: ("right", "fire"),
    12: ("left", "fire"),
    13: ("down", "fire"),
    14: ("up", "right", "fire"),
    15: ("up", "left", "fire"),
    16: ("down", "right", "fire"),
    17: ("down", "left", "fire"),
}

primitive_actions = ["noop","fire","up","right","left","down"]
train_action_files = {action: [[], []] for action in primitive_actions}
test_action_files = {action: [[], []] for action in primitive_actions}
train_pos_weights = {action: [] for action in primitive_actions}
train_neg_weights = {action: [] for action in primitive_actions}

# Script for writing train action files
for _, row in train.iterrows():
  s_id = "s" + str(row["frameid"].replace("_",""))
  action_code = row['action']
  action_name = actions.get(action_code)
  example_weight = row.get('example_weight', 1.0)
  taken_actions = []
  if isinstance(action_name, tuple):
    taken_actions = list(action_name)
  else:
    taken_actions = [action_name]
  
  for pa in primitive_actions:
    action_str = "action(" + s_id + f", {pa})."
    action_str = action_str.lower()
    
    if pa in taken_actions:
      train_action_files[pa][0].append(action_str)
    else:
      for other_action in primitive_actions:
        if other_action != pa:
          other_action_str = "action(" + s_id + f", {other_action})."
          other_action_str = other_action_str.lower()
          train_action_files[pa][1].append(other_action_str)

# Write train files
for pa in primitive_actions:
  with open(f"{base_dir}/"+pa+"/train/"+f"train_pos.txt", "w") as f:
    f.write("\n".join(train_action_files[pa][0]))
  with open(f"{base_dir}/"+pa+"/train/"+f"train_neg.txt", "w") as f:
    f.write("\n".join(train_action_files[pa][1]))

# Script for writing test action files
for _, row in test.iterrows():
  s_id = "s" + str(row["frameid"].replace("_",""))
  action_code = row['action']
  action_name = actions.get(action_code)
  taken_actions = []
  if isinstance(action_name, tuple):
    taken_actions = list(action_name)
  else:
    taken_actions = [action_name]
  
  for pa in primitive_actions:
    action_str = "action(" + s_id + f", {pa})."
    action_str = action_str.lower()
    
    if pa in taken_actions:
      test_action_files[pa][0].append(action_str)
    else:
      test_action_files[pa][1].append(action_str)
      # for other_action in primitive_actions:
      #   if other_action != pa:
      #     other_action_str = "action(" + s_id + f", {other_action})."
      #     other_action_str = other_action_str.lower()
      #     test_action_files[pa][1].append(other_action_str)

# Write test files
for pa in primitive_actions:
  with open(f"{base_dir}/"+pa+"/test/"+f"test_pos.txt", "w") as f:
    f.write("\n".join(test_action_files[pa][0]))
  with open(f"{base_dir}/"+pa+"/test/"+f"test_neg.txt", "w") as f:
    f.write("\n".join(test_action_files[pa][1]))

# Script for writing facts

# Privileged facts contain inRadius if fact_weights for that fact is greater than THRESHOLD

# Fact weights file should be formatted as ground_fact(state, args). weight \n for each fact. Facts are the r
for pa in primitive_actions:
  
  with open(f"{base_dir}/"+pa+"/train/"+'train_facts.txt', 'w') as f:
    with open(f"{base_dir}/"+pa+"/train/"+'fact_weights.txt', 'w') as f2:
      with open(f"{base_dir}/" + pa + "/train/" + "train_facts_pi.txt", "w") as f3:
      
        prev_state = ""
        prev_objnum = ""
        for _, row in train.iterrows():
          rels = str(row["relationships"])
          weights = str(row["predicate_weights"])
          weights_list = []
          if weights != "nan":
            weights_list = weights.split(" ")
          
          if weights_list:
            if rels != "nan":
              s_id = "s" + str(row["frameid"].replace("_",""))
              rels = rels.split(" , ")
              for (rel, w) in zip(rels, weights_list):
                rel = rel.strip()
                if rel:
                  if "(" not in rel:
                    rel = rel+"()"
                  rel = rel.replace("(","(" + s_id + ",")
                  rel = rel.replace(",)", ")")
                  if not rel.endswith("."):
                    rel += "."
                  rel = rel.lower()
                  rel = rel.replace("_", "")
                  f.write(rel + "\n")
                  f2.write(rel + " " + w + "\n")
                  
                  # If visible in rel ignore
                  if "visible" in rel:
                    continue

                  # If visible not in rel and rel has a state and a object arg in the paranthesis, store it and create a rel inradiusobjtype(state, objnum)
                  if "visible" not in rel and rel.count(",") == 1:
                    
                    state, obj_num = rel.split("(")[1].split(",")
                    obj_num = obj_num.split(")")[0]
                    if prev_state == state and prev_objnum == obj_num:
                      if float(w) > THRESHOLD:
                        f3.write(rel + "\n")
                    else:
                      prev_state = state
                      prev_objnum = obj_num
                      objtype = obj_num.split(")")[0][:-1]
                      if float(w) > THRESHOLD:
                        f3.write("inradius" + objtype + "(" + state + "," + obj_num + ").\n")
                        f3.write(rel + "\n")

# For all actions open train_facts_pi.txt and remove duplicate inRadius lines with the same state number and objectnumber
for pa in primitive_actions:
  # Remove duplicate inradius lines from train_facts_pi.txt
  pi_file_path = f"{base_dir}/{pa}/train/train_facts_pi.txt"
  if os.path.exists(pi_file_path):
    with open(pi_file_path, 'r') as f_pi:
      lines = f_pi.readlines()
    
    seen_inradius = set()
    new_lines = []
    for line in lines:
      if line.startswith("inradius"):
        if line in seen_inradius:
          continue
        seen_inradius.add(line)
      # print(line)
      new_lines.append(line)
    
    with open(pi_file_path, 'w') as f_pi:
      f_pi.writelines(new_lines)

  with open(f"{base_dir}/"+pa+"/test/"+'test_facts.txt', 'w') as f:
    for _, row in test.iterrows():
      rels = str(row["relationships"])
      if rels != "nan":
        s_id = "s" + str(row["frameid"].replace("_",""))
        for rel in rels.split(" , "):
          rel = rel.strip()
          if rel:
            if "(" not in rel:
              rel = rel+"()"
            rel = rel.replace("(","(" + s_id + ",")
            rel = rel.replace(",)", ")")
            if not rel.endswith("."):
              rel += "."
            rel = rel.lower()
            rel = rel.replace("_", "")
            f.write(rel + "\n")

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
      if line and '\t' in line:
        parts = line.split('\t')
        if len(parts) >= 2:
          fact = parts[0]
          weight = parts[1].replace('.', '')
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
  print("Removing facts with 0.00 weights from train_facts.txt files...")
  for pa in primitive_actions:
    train_facts_file = f"{base_dir}/{pa}/train/train_facts.txt"
    weights_file = f"{base_dir}/{pa}/train/fact_weights.tsv"
    remove_zero_weight_facts(train_facts_file, weights_file)

with open(f"{base_dir}/"+pa+"/train/"+'train_facts.txt', 'r') as f:
  train_facts = f.read().splitlines()
with open(f"{base_dir}/"+pa+"/test/"+'test_facts.txt', 'r') as f:
  test_facts = f.read().splitlines()

train_dict = {}
test_dict = {}
bk_dict = {}
unique_states = []
for action in primitive_actions:
  train_dict[action]= Database()
  test_dict[action]= Database()
  bk_dict[action] = Background()
  mode = modes.copy()

  train_action_files[action][0] = [s.translate({ord(c):None for c in string.whitespace}) for s in train_action_files[action][0]]
  train_action_files[action][1] = [s.translate({ord(c):None for c in string.whitespace}) for s in train_action_files[action][1]]
  test_action_files[action][0] = [s.translate({ord(c):None for c in string.whitespace}) for s in test_action_files[action][0]]
  test_action_files[action][1] = [s.translate({ord(c):None for c in string.whitespace}) for s in test_action_files[action][1]]
  train_facts = [s.translate({ord(c):None for c in string.whitespace}) for s in train_facts]
  test_facts = [s.translate({ord(c):None for c in string.whitespace}) for s in test_facts]

  train_dict[action].pos = train_action_files[action][0]
  train_dict[action].neg = train_action_files[action][1]
  test_dict[action].pos = test_action_files[action][0]
  test_dict[action].neg = test_action_files[action][1]
  train_dict[action].facts = train_facts
  test_dict[action].facts = test_facts

  bk_dict[action] = Background(modes=mode, bridgers=bridgers, number_of_clauses=20,number_of_cycles=20)
  with open(f"{base_dir}/"+action+"/train/"+f"train_bk.txt", "w") as f:
    f.write(str(bk_dict[action]))
  with open(f"{base_dir}/"+action+"/test/"+f"test_bk.txt", "w") as f:
    f.write(str(bk_dict[action]))
