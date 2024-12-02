import gym
import numpy as np
# from itertools import count
# from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time



# Replace argparse with direct configuration
GAMMA = 0.99  # discount factor
SEED = 543    # random seed
RENDER = False  # whether to render the environment
LOG_INTERVAL = 10  # logging frequency


# Replace the env creation line (around line 24) with:
env = gym.make(
    'CartPole-v1', 
    # render_mode="human"
)
env.reset(seed=SEED)
torch.manual_seed(SEED)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []  # Replace deque with list
    
    # Calculate returns in reverse order
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)  # Insert at beginning instead of appendleft
        
    # Convert to tensor and normalize
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    # Calculate policy loss
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    # Update policy
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
    # Clear episode data
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    start = time.time()
    running_reward = 10
    for i_episode in range(1, 1000000):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if RENDER:
            # if True:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % LOG_INTERVAL == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()