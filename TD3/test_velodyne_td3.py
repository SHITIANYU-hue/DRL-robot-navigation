import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename),map_location=device)
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
eva_steps=1000
eva_eps=3

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()
DELAY=0.3
rewards=[]
run_steps=[]
distances=[]
agl_distances=[]
linear_vs=[]
lateral_vs=[]
# Begin the testing loop
for i in range(eva_eps):
    reward_eps=0
    run_step=0
    distance=0
    agl=0
    linear_v=0
    lateral_v=0
    max_dist=0
    for j in range(eva_steps):
        action = network.get_action(np.array(state))
        # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
        a_in = [(action[0] + 1) / 2, action[1]]
        next_state, reward, done, target,info = env.step(a_in) ## info[distance to goal, angle to goal, linerar vel, angular vel]
        if info[0]>max_dist:
            max_dist=info[0]
        print('max dis',max_dist)
        done = 1 if episode_timesteps + 1 == max_ep else int(done)
        # On termination of episode
        if done:
            state = env.reset()
            done = False
            episode_timesteps = 0
            break
        else:
            state = next_state
            episode_timesteps += 1
        reward_eps+=reward
        run_step+=1
        distance+= DELAY*abs(info[2])
        agl+=DELAY*abs(info[3])
        linear_v+=abs(info[2])
        lateral_v+=abs(info[3])
    rewards.append(reward_eps)
    run_steps.append(run_step)
    distances.append(distance)
    agl_distances.append(agl)
    linear_vs.append(linear_v/run_step)
    lateral_vs.append(lateral_v/run_step)
    # print('eps:',i,'eps reward:',reward_eps)
    # print('run steps:',run_step)
print('avg reward',np.mean(rewards))
print('avg run steps',np.mean(run_steps))
print('avg longitudinal distance',np.mean(distance))
print('avg lateral distance',np.mean(agl_distances))
print('avg linearl vel',np.mean(linear_vs))
print('avg lateral vel',np.mean(lateral_vs))
