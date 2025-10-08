import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Process relationship file")
parser.add_argument("--file", type=str, default="", help="Relationship file")
args = parser.parse_args()

print(args.file)

df = pd.read_csv(args.file, low_memory=False)

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
  frame_id = str(row["frameid"]).replace("_","").lower()
  s_id = "s" + str(frame_id)
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
    with open(f"train_positive_{pa}.txt", "w") as f:
      f.write("\n".join(train_action_files[pa][0]))
    with open(f"train_negative_{pa}.txt", "w") as f:
      f.write("\n".join(train_action_files[pa][1]))

# Remove rows which have action>5

for _, row in test.iterrows():
  frame_id = str(row["frameid"]).replace("_","").lower()
  s_id = "s" + str(frame_id)
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
    with open(f"test_positive_{pa}.txt", "w") as f:
      f.write("\n".join(test_action_files[pa][0]))
    with open(f"test_negative_{pa}.txt", "w") as f:
      f.write("\n".join(test_action_files[pa][1]))

#Script for writing facts
with open('train_facts.txt', 'w') as f:
  for _, row in train.iterrows():
    rels = str(row["relationships"])
    if rels != "nan":
      frame_id = str(row["frameid"]).replace("_","").lower()
      s_id = "s" + str(frame_id)
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

with open('test_facts.txt', 'w') as f:
  for _, row in test.iterrows():
    rels = str(row["relationships"])
    if rels != "nan":
      frame_id = str(row["frameid"]).replace("_","").lower()
      s_id = "s" + str(frame_id)
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



print(test.shape)
total_test = 0
for action in primitive_actions:
  total_test += len(test_action_files[action][0])

print(total_test)
