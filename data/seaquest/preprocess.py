import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from srlearn import Database, Background
import string
import os

horizontal_actions = ["left", "right"]
vertical_actions = ["up", "down"]
fire_actions = ["fire"]

def write_facts_file(df,file_path):
  with open(file_path, 'w') as f:
    for _, row in df.iterrows():
        rels = str(row["relationships"])
        if rels != "nan":
          # frame_id = str(row["frameid"]).split("_")[-1]
          # # s_id = "s" + str(frame_id)
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
"visibleMissile(+state, -missile)."

]

bridgers = ["vissibleMissile/2",
"vissibleEnemy/2",
"vissibleEnemySubmarine/2",
"vissibleDiver/2"]

primitive_actions = ["noop","fire","up","right","left","down"]
actions_classes = ["horizontal","vertical","fire"]

bridgers = [bridger.lower() for bridger in bridgers]

modes = [mode.lower() for mode in modes]

base_dir = "data/seaquest"

for action in horizontal_actions:
  #delete the directory if it exists
  if os.path.exists(f"{base_dir}/horizontal/{action}"):
    import shutil
    shutil.rmtree(f"{base_dir}/horizontal/{action}")
  #create the directory
  os.makedirs(f"{base_dir}/horizontal/{action}/train", exist_ok=True)
  os.makedirs(f"{base_dir}/horizontal/{action}/test", exist_ok=True)
for action in vertical_actions:
  #delete the directory if it exists
  if os.path.exists(f"{base_dir}/vertical/{action}"):
    import shutil
    shutil.rmtree(f"{base_dir}/vertical/{action}")
  #create the directory
  os.makedirs(f"{base_dir}/vertical/{action}/train", exist_ok=True)
  os.makedirs(f"{base_dir}/vertical/{action}/test", exist_ok=True)
for action in fire_actions:
  #delete the directory if it exists
  if os.path.exists(f"{base_dir}/fire/{action}"):
    import shutil
    shutil.rmtree(f"{base_dir}/fire/{action}")
  #create the directory
  os.makedirs(f"{base_dir}/fire/{action}/train", exist_ok=True)
  os.makedirs(f"{base_dir}/fire/{action}/test", exist_ok=True)

# for cl in actions_classes:
#   if os.path.exists(f"{base_dir}/{cl}"):
#     import shutil
#     shutil.rmtree(f"{base_dir}/{cl}")
#   os.makedirs(f"{base_dir}/{cl}/train", exist_ok=True)
#   os.makedirs(f"{base_dir}/{cl}/test", exist_ok=True)

# for action in primitive_actions:
#   #delete the directory if it exists
#   if os.path.exists(f"{base_dir}/{action}"):
#     import shutil
#     shutil.rmtree(f"{base_dir}/{action}")
#   #create the directory
#   os.makedirs(f"{base_dir}/{action}/train", exist_ok=True)
#   os.makedirs(f"{base_dir}/{action}/test", exist_ok=True)

parser = argparse.ArgumentParser(description="Process relationship file")
parser.add_argument("--file", type=str, default="", help="Relationship file")
parser.add_argument("--node_size", type=str, default=2, help="Node size for background")
parser.add_argument("--max_tree_depth", type=str, default=3, help="Max tree depth for background")
parser.add_argument("--remove_0_weights", action="store_true", help="Remove facts with 0.00 weights from train_facts.txt using fact_weights.tsv")
args = parser.parse_args()

print(args.file)

df = pd.read_csv(args.file)


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

train_action_files = {action: [[], []] for action in primitive_actions}
test_action_files = {action: [[], []] for action in primitive_actions}




horizontal_train_action_files = {action: [[], []] for action in horizontal_actions}  # [positive, negative]
vertical_train_action_files = {action: [[], []] for action in vertical_actions} # [positive, negative]
fire_train_action_files = {'fire':[[], []]} # [positive, negative]

for _,row in train.iterrows():
  s_id = "s" + str(row["frameid"].replace("_",""))
  action_code = row['action']
  action_name = actions.get(action_code)
  taken_actions = []
  if isinstance(action_name, tuple):
    taken_actions = list(action_name)
  else:
    taken_actions = [action_name]
  #Append in 0 list if the action is taken else in 1 list for each of horizontal, vertical and fire actions
  # h_count = 0
  for ha in horizontal_actions:
    if ha in taken_actions:
      horizontal_train_action_files[ha][0].append(f"horizontal_{ha}({s_id}).".lower())
    else:
      horizontal_train_action_files[ha][1].append(f"horizontal_{ha}({s_id}).".lower())
  
  for va in vertical_actions:
    if va in taken_actions:
      vertical_train_action_files[va][0].append(f"vertical_{va}({s_id}).".lower())
    else:
      vertical_train_action_files[va][1].append(f"vertical_{va}({s_id}).".lower())
  
  for fa in fire_actions:
    if fa in taken_actions:
      fire_train_action_files[fa][0].append(f"fire_{fa}({s_id}).".lower())
    else:
      fire_train_action_files[fa][1].append(f"fire_{fa}({s_id}).".lower())
 


  #Write the horizontal, vertical and fire action files
  for ha in horizontal_actions:
    with open(f"{base_dir}/horizontal/{ha}/train/"+f"train_pos.txt", "w") as f:
      f.write("\n".join(horizontal_train_action_files[ha][0]) + "\n")
    with open(f"{base_dir}/horizontal/{ha}/train/"+f"train_neg.txt", "w") as f:
      f.write("\n".join(horizontal_train_action_files[ha][1]) + "\n")
  for va in vertical_actions:
    with open(f"{base_dir}/vertical/{va}/train/"+f"train_pos.txt", "w") as f:
      f.write("\n".join(vertical_train_action_files[va][0]) + "\n")
    with open(f"{base_dir}/vertical/{va}/train/"+f"train_neg.txt", "w") as f:
      f.write("\n".join(vertical_train_action_files[va][1]) + "\n")
  for fa in fire_actions:
    with open(f"{base_dir}/fire/fire/train/"+f"train_pos.txt", "w") as f:
      f.write("\n".join(fire_train_action_files[fa][0]) + "\n")
    with open(f"{base_dir}/fire/fire/train/"+f"train_neg.txt", "w") as f:
      f.write("\n".join(fire_train_action_files[fa][1]) + "\n")

horizontal_test_action_files = {action: [[], []] for action in horizontal_actions}  # [positive, negative]
vertical_test_action_files = {action: [[], []] for action in vertical_actions} # [positive, negative]
fire_test_action_files = {'fire':[[], []]} # [positive, negative]

for _,row in test.iterrows():
  s_id = "s" + str(row["frameid"].replace("_",""))
  action_code = row['action']
  action_name = actions.get(action_code)
  taken_actions = []
  if isinstance(action_name, tuple):
    taken_actions = list(action_name)
  else:
    taken_actions = [action_name]
  

  for ha in horizontal_actions:
    if ha in taken_actions:
      horizontal_test_action_files[ha][0].append(f"horizontal_{ha}({s_id}).".lower())
    else:
      horizontal_test_action_files[ha][1].append(f"horizontal_{ha}({s_id}).".lower())
  
  for va in vertical_actions:
    if va in taken_actions:
      vertical_test_action_files[va][0].append(f"vertical_{va}({s_id}).".lower())
    else:
      vertical_test_action_files[va][1].append(f"vertical_{va}({s_id}).".lower())
  
  for fa in fire_actions:
    if fa in taken_actions:
      fire_test_action_files[fa][0].append(f"fire_{fa}({s_id}).".lower())
    else:
      fire_test_action_files[fa][1].append(f"fire_{fa}({s_id}).".lower())

  #Write the horizontal, vertical and fire action files
  for ha in horizontal_actions:
    with open(f"{base_dir}/horizontal/{ha}/test/"+f"test_pos.txt", "w") as f:
      f.write("\n".join(horizontal_test_action_files[ha][0]) + "\n")
    with open(f"{base_dir}/horizontal/{ha}/test/"+f"test_neg.txt", "w") as f:
      f.write("\n".join(horizontal_test_action_files[ha][1]) + "\n")
  for va in vertical_actions:
    with open(f"{base_dir}/vertical/{va}/test/"+f"test_pos.txt", "w") as f:
      f.write("\n".join(vertical_test_action_files[va][0]) + "\n")
    with open(f"{base_dir}/vertical/{va}/test/"+f"test_neg.txt", "w") as f:
      f.write("\n".join(vertical_test_action_files[va][1]) + "\n")
  for fa in fire_actions:
    with open(f"{base_dir}/fire/fire/test/"+f"test_pos.txt", "w") as f:
      f.write("\n".join(fire_test_action_files[fa][0]) + "\n")
    with open(f"{base_dir}/fire/fire/test/"+f"test_neg.txt", "w") as f:
      f.write("\n".join(fire_test_action_files[fa][1]) + "\n")

#Script for writing facts
for action in horizontal_actions:
  write_facts_file(train, f"{base_dir}/horizontal/"+action+"/train/"+'train_facts.txt')
  write_facts_file(test, f"{base_dir}/horizontal/"+action+"/test/"+'test_facts.txt')
for action in vertical_actions:
  write_facts_file(train, f"{base_dir}/vertical/"+action+"/train/"+'train_facts.txt')
  write_facts_file(test, f"{base_dir}/vertical/"+action+"/test/"+'test_facts.txt')
for action in fire_actions:
  write_facts_file(train, f"{base_dir}/fire/"+action+"/train/"+'train_facts.txt')
  write_facts_file(test, f"{base_dir}/fire/"+action+"/test/"+'test_facts.txt')
    
# for cl in actions_classes:
#   with open(f"{base_dir}/"+cl+"/train/"+'train_facts.txt', 'w') as f:
#     for _, row in train.iterrows():
#       rels = str(row["relationships"])
#       if rels != "nan":
#         # frame_id = str(row["frameid"]).split("_")[-1]
#         # # s_id = "s" + str(frame_id)
#         s_id = "s" + str(row["frameid"].replace("_",""))
#         for rel in rels.split(" , "):
#           rel = rel.strip()
#           if rel:
#             if "(" not in rel:
#               rel = rel+"()"
#             rel = rel.replace("(","(" + s_id + ",")
#             rel = rel.replace(",)", ")")
#             if not rel.endswith("."):
#               rel += "."

#             rel = rel.lower()
#             rel = rel.replace("_", "")
#             f.write(rel + "\n")

#   with open(f"{base_dir}/"+cl+"/test/"+'test_facts.txt', 'w') as f:
#     for _, row in test.iterrows():
#       rels = str(row["relationships"])
#       if rels != "nan":
#         # frame_id = str(row["frameid"]).split("_")[-1]
#         # # s_id = "s" + str(frame_id)
#         s_id = "s" + str(row["frameid"].replace("_",""))
#         for rel in rels.split(" , "):
#           rel = rel.strip()
#           if rel:
#             if "(" not in rel:
#               rel = rel+"()"
#             rel = rel.replace("(","(" + s_id + ",")
#             rel = rel.replace(",)", ")")
#             if not rel.endswith("."):
#               rel += "."

#             rel = rel.lower()
#             rel = rel.replace("_", "")
#             f.write(rel + "\n")


with open(f"{base_dir}/horizontal/left/train/"+'train_facts.txt', 'r') as f:
  train_facts = f.read().splitlines()
with open(f"{base_dir}/horizontal/left/test/"+'test_facts.txt', 'r') as f:
  test_facts = f.read().splitlines()

train_dict = {}
test_dict = {}
bk_dict = {}
unique_states = []

train_dict["horizontal"]= Database()
test_dict["horizontal"]= Database()
train_dict["horizontal"].pos = horizontal_train_action_files[ha][0]
train_dict["horizontal"].neg = horizontal_train_action_files[ha][1]
test_dict["horizontal"].pos = horizontal_test_action_files[ha][0]
test_dict["horizontal"].neg = horizontal_test_action_files[ha][1]
train_dict["horizontal"].facts = train_facts
test_dict["horizontal"].facts = test_facts

train_dict["vertical"]= Database()
test_dict["vertical"]= Database()
train_dict["vertical"].pos = vertical_train_action_files[va][0]
train_dict["vertical"].neg = vertical_train_action_files[va][1]
test_dict["vertical"].pos = vertical_test_action_files[va][0]
test_dict["vertical"].neg = vertical_test_action_files[va][1]
train_dict["vertical"].facts = train_facts
test_dict["vertical"].facts = test_facts

train_dict["fire"]= Database()
test_dict["fire"]= Database()
train_dict["fire"].pos = fire_train_action_files[fa][0]
train_dict["fire"].neg = fire_train_action_files[fa][1]
test_dict["fire"].pos = fire_test_action_files[fa][0]
test_dict["fire"].neg = fire_test_action_files[fa][1]
train_dict["fire"].facts = train_facts
test_dict["fire"].facts = test_facts


for cl in ["horizontal_left", "horizontal_right", "vertical_up","vertical_down", "fire_fire"]:
  bk_dict[cl] = Background()
  mode = modes.copy()
  mode.append(f"{cl}(+state).")

  bk_dict[cl] = Background(modes=mode, bridgers=bridgers, number_of_clauses=20,number_of_cycles=20, node_size=int(args.node_size), 
                               max_tree_depth=int(args.max_tree_depth))
  #write the background to file
  with open(f"{base_dir}/"+cl.replace("_","/")+"/train/"+f"train_bk.txt", "w") as f:
    f.write(str(bk_dict[cl]))
  with open(f"{base_dir}/"+cl.replace("_","/")+"/test/"+f"test_bk.txt", "w") as f:
    f.write(str(bk_dict[cl]))
