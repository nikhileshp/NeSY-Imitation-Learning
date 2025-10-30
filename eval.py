import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
import argparse

parser = argparse.ArgumentParser(description="Process relationship file")
parser.add_argument("--model_dir", type=str, default="", help="Model directory path")

args = parser.parse_args()

primitive_actions = ["horizontal_left", "horizontal_right", "vertical_up", "vertical_down", "fire_fire"]

state_ids = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
  test_query_file = f"data/seaquest/{action.replace("_","/")}/test/query_{action}.db"
  
  with open(test_query_file, "r") as f:
    lines = f.read().splitlines()
    for line in lines:
        state_id = line.split("(")[1].split(")")[0]
        if "!" in line:
            state_ids[action][1].append(state_id)
        else:
            state_ids[action][0].append(state_id)


pred_prob = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
  auc_file = f"data/seaquest/{action.replace("_","/")}/test/AUC/aucTemp.txt"
  with open(auc_file, "r") as f:
    lines = f.read().splitlines()
    for i,line in enumerate(lines):
        parts = line.split()
        if i < len(state_ids[action][0]):
           pred_prob[action][0].append(float(parts[0]))
        else:
           pred_prob[action][1].append(float(parts[0]))

dict_action_prob = {"noop":{}, "fire":{}, "up":{}, "right":{}, "left":{}, "down":{}}



state_id_list = [state_id for action in primitive_actions for state_id in state_ids[action][0]] + \
                [state_id for action in primitive_actions for state_id in state_ids[action][1]]

state_id_action_probs = {state_id: [] for state_id in state_id_list}

for action in primitive_actions:
    for i, state_id in enumerate(state_ids[action][0]):
        state_id_action_probs[state_id].append(pred_prob[action][0][i])
    for i, state_id in enumerate(state_ids[action][1]):
        state_id_action_probs[state_id].append(pred_prob[action][1][i])


for action in primitive_actions:
    for i, state_id in enumerate(state_ids[action][0]):
        dict_action_prob[action][state_id] = pred_prob[action][0][i]
    for i, state_id in enumerate(state_ids[action][1]):
        dict_action_prob[action][state_id] = pred_prob[action][1][i]


# Method 1: Direct argmax (original method)
max_prob_index = {state_id: np.argmax(probs) for state_id, probs in state_id_action_probs.items()}

# Method 2: Softmax normalized probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x)

max_prob_index_normalized = {}
for state_id, probs in state_id_action_probs.items():
    normalized_probs = softmax(np.array(probs))
    max_prob_index_normalized[state_id] = np.argmax(normalized_probs)

test_df = pd.read_csv("test.csv")
test_df['state_id'] = test_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))
print(test_df.head())
state_ids = sorted(list(test_df['state_id']))
print(state_id_list[0:10])
print('sRZ80297210013' in state_id_list)

# Predict using both methods
test_df['predicted_action'] = test_df['state_id'].apply(lambda x: max_prob_index[x])
test_df['predicted_action_normalized'] = test_df['state_id'].apply(lambda x: max_prob_index_normalized[x])


# Compare results
print("\n" + "="*80)
print("METHOD 1: Direct argmax (non-normalized probabilities)")
print("="*80)
print(classification_report(test_df['action'], test_df['predicted_action'], target_names=primitive_actions))
print("\nConfusion Matrix:")
print(confusion_matrix(test_df['action'], test_df['predicted_action']))

print("\n" + "="*80)
print("METHOD 2: Softmax normalized probabilities")
print("="*80)
print(classification_report(test_df['action'], test_df['predicted_action_normalized'], target_names=primitive_actions))
print("\nConfusion Matrix:")
print(confusion_matrix(test_df['action'], test_df['predicted_action_normalized']))

# Check if predictions differ
num_different = (test_df['predicted_action'] != test_df['predicted_action_normalized']).sum()
print(f"\n{'='*80}")
print(f"Number of states where predictions differ: {num_different} / {len(test_df)}")
print(f"Percentage of different predictions: {100 * num_different / len(test_df):.2f}%")
print("="*80)

#write the classification report to a file
with open(f"{args.model_dir}/eval_report.txt", "w") as f:
    f.write("METHOD 1: Direct argmax (non-normalized probabilities)\n")
    f.write("="*80 + "\n")
    f.write(classification_report(test_df['action'], test_df['predicted_action'], target_names=primitive_actions))
    f.write("\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(test_df['action'], test_df['predicted_action'])))
    f.write("\n\n")
    f.write("METHOD 2: Softmax normalized probabilities\n")
    f.write("="*80 + "\n")
    f.write(classification_report(test_df['action'], test_df['predicted_action_normalized'], target_names=primitive_actions))
    f.write("\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(test_df['action'], test_df['predicted_action_normalized'])))
    f.write(f"\n\nNumber of states where predictions differ: {num_different} / {len(test_df)}\n")
    f.write(f"Percentage of different predictions: {100 * num_different / len(test_df):.2f}%\n")


# test_file = f"data/seaquest/noop/test/test_pos.txt"
# with open(test_file, "r") as f:
#   lines = f.read().splitlines()
#   for line in lines:
#     state_id = line.split("(")[1].split(")")[0]
#     state_ids.append(state_id)
# pred_prob['state_id'] = state_ids

#convert to numpy array
# for action in primitive_actions:
#   pred_prob[action] = np.array(pred_prob[action])

# pred_prob_df = pd.DataFrame(pred_prob)
# print(pred_prob_df.head())