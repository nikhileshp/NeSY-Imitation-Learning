import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

primitive_actions = ["noop","fire","up","right","left","down"]

state_ids = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
  test_query_file = f"data/seaquest/{action}/test/query_{action}.db"
  
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


max_prob_index = {state_id: np.argmax(probs) for state_id, probs in state_id_action_probs.items()}

test_df = pd.read_csv("test.csv")
test_df['state_id'] = test_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))
print(test_df.head())
state_ids = sorted(list(test_df['state_id']))
print(state_id_list[0:10])
print('sRZ80297210013' in state_id_list)
test_df['predicted_action'] = test_df['state_id'].apply(lambda x: max_prob_index[x])


#Compute weighted f1 score from test_df


print(classification_report(test_df['action'], test_df['predicted_action'], target_names=primitive_actions))
print(confusion_matrix(test_df['action'], test_df['predicted_action']))

#write the classification report to a file
with open("eval_report.txt", "w") as f:
    f.write(classification_report(test_df['action'], test_df['predicted_action'], target_names=primitive_actions))
    f.write("\n")
    f.write(str(confusion_matrix(test_df['action'], test_df['predicted_action'])))


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