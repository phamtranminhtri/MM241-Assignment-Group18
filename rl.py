import gymnasium as gym
import gym_cutting_stock
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
SEED = 42    # random seed
RENDER = False  # whether to render the environment
LOG_INTERVAL = 10  # logging frequency


# Replace the env creation line (around line 24) with:
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0", 
    # render_mode="human"
)
NUM_EPISODES = 100
observation, info = env.reset(seed=SEED)
torch.manual_seed(SEED)

num_stocks = len(observation["stocks"])
first_stock = observation["stocks"][0]
max_h, max_w = first_stock.shape
num_products = len(observation["products"])


# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.dropout = nn.Dropout(p=0.6)
#         self.affine2 = nn.Linear(128, 2)

#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         x = self.affine1(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         action_scores = self.affine2(x)
#         return F.softmax(action_scores, dim=1)


# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
        
#         # Convolutional layers for stock grids
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
#         # Fully connected layers
#         conv_output_size = 128 * max_h * max_w
#         self.fc1 = nn.Linear(conv_output_size, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.dropout = nn.Dropout(p=0.3)
        
#         # Output layers for actions
#         self.stock_head = nn.Linear(128, num_stocks)
#         self.size_head = nn.Linear(128, 2)       # Outputs [width, height]
#         self.position_head = nn.Linear(128, 2)   # Outputs [x, y]
        
#         # Optional: Learnable log standard deviations for size and position
#         self.size_log_std = nn.Parameter(torch.zeros(2))
#         self.position_log_std = nn.Parameter(torch.zeros(2))
        
#         # Storage for training
#         self.saved_log_probs = []
#         self.rewards = []
    
#     def forward(self, state):
#         # Process stocks
#         stocks = state['stocks']
#         stocks_tensor = torch.stack([
#             self.preprocess_stock(torch.tensor(s, dtype=torch.float32))
#             for s in stocks
#         ])  # Shape: [num_stocks, max_h, max_w]
#         stocks_tensor = stocks_tensor.unsqueeze(1)  # Add channel dimension
        
#         # Forward pass through CNN
#         x = F.relu(self.conv1(stocks_tensor))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
        
#         # Flatten features
#         x = x.view(x.size(0), -1)  # Shape: [num_stocks, conv_output_size]
        
#         # Aggregate features (e.g., mean over stocks)
#         x = x.mean(dim=0, keepdim=True)  # Shape: [1, conv_output_size]
        
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        
#         # Output action components
#         stock_logits = self.stock_head(x)           # Shape: [1, num_stocks]
#         size_mean = self.size_head(x)               # Shape: [1, 2]
#         position_mean = self.position_head(x)       # Shape: [1, 2]
        
#         return stock_logits, size_mean, position_mean

#     def preprocess_stock(self, stock):
#         # Map cell values to suitable numeric values for CNN input
#         stock_processed = stock.clone()
#         stock_processed[stock_processed == -2] = 0.0    # Unavailable cells
#         stock_processed[stock_processed == -1] = 1.0    # Available cells
#         stock_processed[stock_processed >= 0] = -1.0    # Occupied cells
#         return stock_processed  # Shape: [max_h, max_w]
    
    
# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
        
#         # Reduced CNN layers (fewer filters)
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 32 -> 16 filters
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 64 -> 32 filters
#         # Removed conv3 layer
        
#         # Reduced fully connected layers
#         conv_output_size = 32 * max_h * max_w  # Updated for fewer filters
#         self.fc1 = nn.Linear(conv_output_size, 128)  # 256 -> 128 units
#         self.fc2 = nn.Linear(128, 64)  # 128 -> 64 units
#         self.dropout = nn.Dropout(p=0.3)
        
#         # Output layers (unchanged)
#         self.stock_head = nn.Linear(64, num_stocks)
#         self.size_head = nn.Linear(64, 2)
#         self.position_head = nn.Linear(64, 2)
        
#         # Parameters (unchanged)
#         self.size_log_std = nn.Parameter(torch.zeros(2))
#         self.position_log_std = nn.Parameter(torch.zeros(2))
        
#         # Storage (unchanged)
#         self.saved_log_probs = []
#         self.rewards = []
    
#     def forward(self, state):
#         # Process stocks
#         stocks = state['stocks']
#         stocks_tensor = torch.stack([
#             self.preprocess_stock(torch.tensor(s, dtype=torch.float32))
#             for s in stocks
#         ])
#         stocks_tensor = stocks_tensor.unsqueeze(1)
        
#         # Simplified CNN forward pass
#         x = F.relu(self.conv1(stocks_tensor))
#         x = F.relu(self.conv2(x))
        
#         # Rest unchanged
#         x = x.view(x.size(0), -1)
#         x = x.mean(dim=0, keepdim=True)
        
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        
#         stock_logits = self.stock_head(x)
#         size_mean = self.size_head(x)
#         position_mean = self.position_head(x)
        
#         return stock_logits, size_mean, position_mean

#     def preprocess_stock(self, stock):
#         stock_processed = stock.clone()
#         stock_processed[stock_processed == -2] = 0.0
#         stock_processed[stock_processed == -1] = 1.0
#         stock_processed[stock_processed >= 0] = -1.0
#         return stock_processed    
    
    
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        
        # Input size: flattened stock grids + product information
        input_size = max_h * max_w * num_stocks + num_products * 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.3)
        
        # Output heads
        self.stock_head = nn.Linear(64, num_stocks)
        self.size_head = nn.Linear(64, 2)
        self.position_head = nn.Linear(64, 2)
        
        # Optional: Learnable log standard deviations for size and position
        self.size_log_std = nn.Parameter(torch.zeros(2))
        self.position_log_std = nn.Parameter(torch.zeros(2))
        
        # Storage for training
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, state):
        # Flatten and concatenate stock grids and product information
        stocks = state['stocks'].reshape(-1)  # Flatten stock grids
        products = state['products'].reshape(-1)  # Flatten product information
        x = torch.cat([stocks, products])
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output actions
        stock_logits = self.stock_head(x)
        size_mean = self.size_head(x)
        position_mean = self.position_head(x)
        
        return stock_logits, size_mean, position_mean

    def preprocess_stock(self, stock):
        stock_processed = stock.clone()
        stock_processed[stock_processed == -2] = 0.0
        stock_processed[stock_processed == -1] = 1.0
        stock_processed[stock_processed >= 0] = -1.0
        return stock_processed    
    
    

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


# def select_action(state):
#     state = torch.from_numpy(state).float().unsqueeze(0)
#     probs = policy(state)
#     m = Categorical(probs)
#     action = m.sample()
#     policy.saved_log_probs.append(m.log_prob(action))
#     return action.item()


# def select_action(state):
#     # Get action components from the policy network
#     stock_logits, size_mean, position_mean = policy(state)
    
#     # Sample stock index
#     stock_dist = Categorical(logits=stock_logits)
#     stock_idx = stock_dist.sample()
#     stock_log_prob = stock_dist.log_prob(stock_idx)
    
#     # Sample size (width, height) from Normal distribution
#     size_std = torch.exp(policy.size_log_std)  # Ensure positive std
#     size_dist = torch.distributions.Normal(size_mean.squeeze(0), size_std)
#     size_sample = size_dist.sample()
#     size_log_prob = size_dist.log_prob(size_sample).sum()
    
#     # Sample position (x, y) from Normal distribution
#     position_std = torch.exp(policy.position_log_std)  # Ensure positive std
#     position_dist = torch.distributions.Normal(position_mean.squeeze(0), position_std)
#     position_sample = position_dist.sample()
#     position_log_prob = position_dist.log_prob(position_sample).sum()
    
#     # Sum log probabilities
#     total_log_prob = stock_log_prob + size_log_prob + position_log_prob
#     policy.saved_log_probs.append(total_log_prob)
    
#     # Convert samples to integers and ensure they are within valid ranges
#     size = size_sample.detach().numpy()
#     size = [int(max(1, min(size[0], max_w))), int(max(1, min(size[1], max_h)))]
    
#     position = position_sample.detach().numpy()
#     position = [
#         int(max(0, min(position[0], max_w - size[0]))),
#         int(max(0, min(position[1], max_h - size[1])))
#     ]
    
#     # Construct the action dictionary
#     action = {
#         'stock_idx': stock_idx.item(),
#         'size': size,                 # [width, height]
#         'position': tuple(position),  # (x, y)
#     }
    
#     return action


def select_action(state):
    # Get action components from the policy network
    stock_logits, size_mean, position_mean = policy(state)
    
    # Sample stock index
    stock_dist = Categorical(logits=stock_logits)
    stock_idx = stock_dist.sample()
    stock_log_prob = stock_dist.log_prob(stock_idx)
    
    # Sample size (width, height) from Normal distribution
    size_std = torch.exp(policy.size_log_std)  # Ensure positive std
    size_dist = torch.distributions.Normal(size_mean, size_std)
    size_sample = size_dist.sample()
    size_log_prob = size_dist.log_prob(size_sample).sum()
    
    # Sample position (x, y) from Normal distribution
    position_std = torch.exp(policy.position_log_std)  # Ensure positive std
    position_dist = torch.distributions.Normal(position_mean, position_std)
    position_sample = position_dist.sample()
    position_log_prob = position_dist.log_prob(position_sample).sum()
    
    # Sum log probabilities
    total_log_prob = stock_log_prob + size_log_prob + position_log_prob
    policy.saved_log_probs.append(total_log_prob)
    
    # Convert samples to integers and ensure they are within valid ranges
    size = size_sample.detach().numpy()
    size = [int(max(1, min(size[0], max_w))), int(max(1, min(size[1], max_h)))]
    
    position = position_sample.detach().numpy()
    position = [
        int(max(0, min(position[0], max_w - size[0]))),
        int(max(0, min(position[1], max_h - size[1])))
    ]
    
    # Construct the action dictionary
    action = {
        'stock_idx': stock_idx.item(),
        'size': size,                 # [width, height]
        'position': tuple(position),  # (x, y)
    }
    
    return action


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
    step = 0
    for i_episode in range(1, NUM_EPISODES):
        print(f"Episode {i_episode}")
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            print(f"Step {step}")
            step += 1
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if RENDER:
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