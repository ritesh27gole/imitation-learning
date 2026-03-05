import pickle
import torch
import pandas as pd
import numpy as np

def get_expert_data(file_path):
    with open(file_path, 'rb') as fp:
        expert_data = pickle.load(fp)
    print('Imported Expert data successfully')
    return expert_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

file_path = 'data/expert_data.pkl'
expert_data = get_expert_data(file_path)
df = pd.DataFrame(expert_data)

print(type(expert_data))
print(df.columns)
print(len(df["observations"][2]))

idxs = np.array(range(len(expert_data)))
episode_length=50
batch_size=32
num_batches = len(idxs)*episode_length // batch_size
np.random.shuffle(idxs)
print(idxs, num_batches)
print(len(expert_data[0]["observations"]))

observations = []
actions = []
for id in idxs:
    observations.append(expert_data[id]["observations"])
    actions.append(expert_data[id]["actions"])
    print(len(observations))
observations = np.array(observations)
actions = np.array(actions)

states = observations.reshape(250, 11)
acts = actions.reshape(250, 2)

print(np.concatenate(states).shape)
print(np.concatenate(acts).shape)

print(list(range(0*batch_size, 1*batch_size)))
print(len(states[list(range(0*batch_size, 1*batch_size))]))
# expert_policy = torch.load('data/expert_policy.pkl', map_location=torch.device(device))
# print("Expert policy loaded")

# print(df.columns)
# observations, next_observations, actions, rewards, dones, images

# observation = [cos(t0), cos(t1), sin(t0), sin(t1), 
#                x_t, y_t, t0_d, t1_d,
#                x-x_t, y-y_t, 0.0]

# 50 such observations are there is each idx 
# i.e. episode and there are 5 such episodes
 
# expert_data[idx] is dict, 
# which has keys: observations, next_observations, actions, rewards, dones, images