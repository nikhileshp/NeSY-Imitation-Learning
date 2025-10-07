import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

primitive_actions = ["noop","fire","up","right","left","down"]

state_ids = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
  pos_test_file = f"data/seaquest/{action}/test/test_pos.txt"
  with open(pos_test_file, "r") as f:
    lines = f.read().splitlines()
    for line in lines:
        state_id = line.split("(")[1].split(")")[0]
        state_ids[action][0].append(state_id)
    neg_test_file = f"data/seaquest/{action}/test/test_neg.txt"
    with open(neg_test_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            state_id = line.split("(")[1].split(")")[0]
            state_ids[action][1].append(state_id)
print(len(state_ids['noop'][0])+len(state_ids['noop'][1]))

pred_prob = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
  auc_file = f"data/seaquest/{action}/test/AUC/aucTemp.txt"
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

state_id_action_probs = {state_id: [] for state_id in state_id_list
                         }

for action in primitive_actions:
    for i, state_id in enumerate(state_ids[action][0]):
        state_id_action_probs[state_id].append(pred_prob[action][0][i])
    for i, state_id in enumerate(state_ids[action][1]):
        state_id_action_probs[state_id].append(pred_prob[action][1][i])

print(state_id_action_probs['s9083']) #list of 6 probs for each action

print(primitive_actions)
for action in primitive_actions:
    for i, state_id in enumerate(state_ids[action][0]):
        dict_action_prob[action][state_id] = pred_prob[action][0][i]
    for i, state_id in enumerate(state_ids[action][1]):
        dict_action_prob[action][state_id] = pred_prob[action][1][i]


max_prob_index = {state_id: np.argmax(probs) for state_id, probs in state_id_action_probs.items()}

test_df = pd.read_csv("test.csv")
test_df['state_id'] = test_df['frameid'].apply(lambda x: "s" + str(x).split("_")[-1])
test_df['predicted_action'] = test_df['state_id'].apply(lambda x: max_prob_index[x])
print(dict_action_prob['right'])

#Compute weighted f1 score from test_df
print(test_df.head())

print(classification_report(test_df['action'], test_df['predicted_action'], target_names=primitive_actions))
print(confusion_matrix(test_df['action'], test_df['predicted_action']))

#{state_id : [prob1, prob2, ...] }
#such that prob1 is the same idx as state_ids


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