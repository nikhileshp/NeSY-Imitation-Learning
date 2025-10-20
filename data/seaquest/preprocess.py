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
"visibleMissile(+state, -missile)."

]

bridgers = ["vissibleMissile/2",
"vissibleEnemy/2",
"vissibleEnemySubmarine/2",
"vissibleDiver/2"]

primitive_actions = ["noop","fire","up","right","left","down"]

bridgers = [bridger.lower() for bridger in bridgers]

modes = [mode.lower() for mode in modes]

for action in primitive_actions:
  #delete the directory if it exists
  if os.path.exists(f"data/seaquest/{action}"):
    import shutil
    shutil.rmtree(f"data/seaquest/{action}")
  #create the directory
  os.makedirs(f"data/seaquest/{action}/train", exist_ok=True)
  os.makedirs(f"data/seaquest/{action}/test", exist_ok=True)

parser = argparse.ArgumentParser(description="Process relationship file")
parser.add_argument("--file", type=str, default="", help="Relationship file")
args = parser.parse_args()

print(args.file)

df = pd.read_csv(args.file)

#remove rows with action greater than 5
df = df[df['action'] <= 5]

# # Create a new column in df that combines the strings in objects and relationships. Call it OR
# df['OR'] = df['objects'] + df['relationships']

# # # For each unique element in df["OR"] list the set of actions
# # # Count the number of unique elements in df["OR"] that has only one corresponding action
# count = 0
# for element in df['OR'].unique():
#     # print(element)
#     if len(df[df['OR'] == element]['action'].unique()) == 1:
#         count += 1
#         # print(df[df['OR'] == element]['action'].unique())
#     # print()


# # For each unique element in df["OR"] which has only one corresponding action remove duplicates
# dframe_unique = df.copy()
# for element in df['OR'].unique():
#     if len(df[df['OR'] == element]['action'].unique()) == 1 :
#       #Remove all other occurrences of element from dframe_unique except 1
#       dframe_unique = dframe_unique[dframe_unique['OR'] != element]

#       # Add that example back so there is a sinlge example for that action
#       dframe_unique = pd.concat([dframe_unique, df[df['OR'] == element].iloc[[0]]])

# #Number of rows that have action greater than 5
# count = 0
# for row in dframe_unique.iterrows():
#   if row[1]['action'] > 5:
#     count += 1
# print(count)

# #Remove rows that have action greater than 5
# dframe_unique = dframe_unique[dframe_unique['action'] <= 5]

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

for _, row in train.iterrows():
  # frame_id = str(row["frameid"]).split("_")[-1]
  # s_id = "s" + str(frame_id)
  s_id = "s" + str(row["frameid"].replace("_",""))
  action_code = row['action']
  action_name = actions.get(action_code)
  taken_actions = []
  if isinstance(action_name, tuple):
    taken_actions = list(action_name)
  else:
    taken_actions = [action_name]
  for pa in primitive_actions:
    action_str = pa + "(" + s_id + ")."
    action_str = action_str.lower()

    if pa in taken_actions:
      train_action_files[pa][0].append(action_str)
    else:
      train_action_files[pa][1].append(action_str)
    with open("data/seaquest/"+pa+"/train/"+f"train_pos.txt", "w") as f:
      f.write("\n".join(train_action_files[pa][0]))
    with open("data/seaquest/"+pa+"/train/"+f"train_neg.txt", "w") as f:
      f.write("\n".join(train_action_files[pa][1]))

# Remove rows which have action>5

for _, row in test.iterrows():
  # frame_id = str(row["frameid"]).split("_")[-1]
  # s_id = "s" + str(frame_id)
  s_id = "s" + str(row["frameid"].replace("_",""))
  action_code = row['action']
  action_name = actions.get(action_code)
  taken_actions = []
  if isinstance(action_name, tuple):
    taken_actions = list(action_name)
  else:
    taken_actions = [action_name]
  for pa in primitive_actions:
    action_str = pa + "(" + s_id + ")."
    action_str = action_str.lower()

    if pa in taken_actions:
      test_action_files[pa][0].append(action_str)
    else:
      test_action_files[pa][1].append(action_str)
    with open("data/seaquest/"+pa+"/test/"+f"test_pos.txt", "w") as f:
      f.write("\n".join(test_action_files[pa][0]))
    with open("data/seaquest/"+pa+"/test/"+f"test_neg.txt", "w") as f:
      f.write("\n".join(test_action_files[pa][1]))

#Script for writing facts
for pa in primitive_actions:
  with open("data/seaquest/"+pa+"/train/"+'train_facts.txt', 'w') as f:
    for _, row in train.iterrows():
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
  
  with open("data/seaquest/"+pa+"/train/"+'fact_weights.tsv', 'w') as f:
    for _, row in train.iterrows():
    
      weights = str(row["distance_weights"])
      weights_list = []
      if weights != "nan":
        state_weights = weights.split(" , ")
        for sw in state_weights:
          frame_id = str(row["frameid"]).replace("_","").lower()
          s_id = "s" + str(frame_id)
          sw = sw.strip()
          if sw:
            if "(" not in sw:
              sw = sw+"()"
            sw = sw.replace("(","(" + s_id + ",")
            sw = sw.replace(",)", ")")
            if not sw.endswith("."):
              sw += "."
            sw = sw.lower()
            sw = sw.replace("_", "")
            sw = sw.replace(" ","\t")
            weights_list.append(sw)
      if weights_list:
        f.write("\n".join(weights_list) + "\n")

  with open("data/seaquest/"+pa+"/test/"+'test_facts.txt', 'w') as f:
    for _, row in test.iterrows():
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

with open("data/seaquest/"+pa+"/train/"+'train_facts.txt', 'r') as f:
  train_facts = f.read().splitlines()
with open("data/seaquest/"+pa+"/test/"+'test_facts.txt', 'r') as f:
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

  mode.append(f"{action}(+state).")

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

  bk_dict[action] = Background(modes=mode, bridgers=bridgers, number_of_clauses=100,number_of_cycles=100)
  #write the background to file
  with open("data/seaquest/"+action+"/train/"+f"train_bk.txt", "w") as f:
    f.write(str(bk_dict[action]))
  with open("data/seaquest/"+action+"/test/"+f"test_bk.txt", "w") as f:
    f.write(str(bk_dict[action]))