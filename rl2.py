import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import gym_cutting_stock
import numpy as np
import gc

# class Policy(nn.Module):

    # def __init__(self, num_stocks, max_width, max_height, num_products):
    #     super(Policy, self).__init__()
        
    #     self.num_stocks = num_stocks
    #     self.max_width = max_width
    #     self.max_height = max_height
    #     self.num_products = num_products
        
    #     # CNN layers for stock processing
    #     self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=2)
    #     self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        
    #     h_out = max_height // 8
    #     w_out = max_width // 8
    #     self.conv_output_size = 4 * h_out * w_out
        
    #     # FC layers
    #     self.product_fc = nn.Linear(num_products * 3, 32)
    #     self.fc1 = nn.Linear(self.conv_output_size + 32, 64)
    #     self.fc2 = nn.Linear(64, 32)
        
    #     # Output heads
    #     self.stock_head = nn.Linear(32, num_stocks)
    #     self.product_head = nn.Linear(32, num_products)  # Product index logits
    #     self.position_head = nn.Linear(32, 2)
        
    #     self.position_log_std = nn.Parameter(torch.zeros(2))
        
    #     self.saved_log_probs = []
    #     self.rewards = []
    
    # def __init__(self, num_stocks, max_width, max_height, num_products):
    #     super(Policy, self).__init__()
        
    #     self.num_stocks = num_stocks
    #     self.max_width = max_width
    #     self.max_height = max_height
    #     self.num_products = num_products
        
    #     # Simplify CNN layers
    #     self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    #     h_out = max_height // 4
    #     w_out = max_width // 4
    #     self.conv_output_size = 2 * h_out * w_out
        
    #     # Reduce FC layers size
    #     self.product_fc = nn.Linear(num_products * 3, 16)
    #     self.fc1 = nn.Linear(self.conv_output_size + 16, 32)
    #     self.fc2 = nn.Linear(32, 16)
        
    #     # Output heads
    #     self.stock_head = nn.Linear(16, num_stocks)
    #     self.product_head = nn.Linear(16, num_products)
    #     self.position_head = nn.Linear(16, 2)
        
    #     self.position_log_std = nn.Parameter(torch.zeros(2))
        
    #     self.saved_log_probs = []
    #     self.rewards = []

    # def process_stock_batch(self, stocks, batch_size=10):
    #     n_stocks = len(stocks)
    #     features = []
        
    #     # Process stocks in batches
    #     for i in range(0, n_stocks, batch_size):
    #         batch_stocks = stocks[i:i + batch_size]
    #         batch_tensor = torch.stack([
    #             self.preprocess_stock(torch.tensor(s, dtype=torch.float32))
    #             for s in batch_stocks
    #         ]).unsqueeze(1)
            
    #         # with torch.no_grad():
    #         x = self.pool(F.relu(self.conv1(batch_tensor)))
    #         x = x.view(len(batch_stocks), -1)
    #         features.append(x)
            
    #         # Clear memory
    #         del batch_tensor
    #         gc.collect()
            
    #     features = torch.cat(features, dim=0)
    #     return features.mean(dim=0, keepdim=True)

    # def forward(self, state):
    #     # Process stocks in batches
    #     x = self.process_stock_batch(state['stocks'])
        
    #     # Process products
    #     products_info = []
    #     for p in state['products']:
    #         products_info.append(np.concatenate([p['size'], [p['quantity']]]))
    #     products_info = np.array(products_info)
        
    #     if len(products_info) < self.num_products:
    #         padding = np.zeros((self.num_products - len(products_info), 3))
    #         products_info = np.vstack([products_info, padding])
        
    #     products_tensor = torch.FloatTensor(products_info).view(1, -1)
    #     p = F.relu(self.product_fc(products_tensor))
        
    #     # Combine features
    #     combined = torch.cat([x, p], dim=1)
    #     combined = F.relu(self.fc1(combined))
    #     combined = F.relu(self.fc2(combined))
        
    #     # Output actions
    #     stock_logits = self.stock_head(combined)
    #     product_logits = self.product_head(combined)  # Changed from size_mean
    #     position_mean = self.position_head(combined)
        
    #     return stock_logits, product_logits, position_mean  # Changed return value order

    # def preprocess_stock(self, stock):
    #     with torch.no_grad():
    #         stock_processed = stock.clone()
    #         stock_processed[stock_processed == -2] = 0.0
    #         stock_processed[stock_processed == -1] = 1.0
    #         stock_processed[stock_processed >= 0] = -1.0
    #     return stock_processed
    
    
# In rl2.py, optimize Policy network
class Policy(nn.Module):
    def __init__(self, num_stocks, max_width, max_height, num_products):
        super(Policy, self).__init__()
        
        self.num_stocks = num_stocks
        self.max_width = max_width
        self.max_height = max_height
        self.num_products = num_products
        
        # Simpler CNN
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=2)
        
        h_out = (max_height - 2) // 2
        w_out = (max_width - 2) // 2
        self.conv_output_size = 2 * h_out * w_out
        
        # Minimal FC layers
        self.fc = nn.Linear(self.conv_output_size + num_products * 3, 32)
        
        # Output heads
        self.stock_head = nn.Linear(32, num_stocks)
        self.product_head = nn.Linear(32, num_products)
        self.position_head = nn.Linear(32, 2)
        
        self.position_log_std = nn.Parameter(torch.ones(2))  # Initialize to 1 for better exploration
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state):
        # Process single stock at a time
        features = []
        for stock in state['stocks']:
            stock_tensor = self.preprocess_stock(torch.tensor(stock, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
            x = F.relu(self.conv1(stock_tensor))
            features.append(x.view(-1))
        
        x = torch.stack(features).mean(dim=0)
        
        # Fix slow tensor creation warning
        products_info = np.array([
            np.concatenate([p['size'], [p['quantity']]])
            for p in state['products']
        ])
        products_tensor = torch.from_numpy(products_info).float().view(-1)
        
        combined = torch.cat([x, products_tensor])
        hidden = F.relu(self.fc(combined))
        
        return (
            self.stock_head(hidden),
            self.product_head(hidden),
            self.position_head(hidden)
        )
        
    def preprocess_stock(self, stock):
        with torch.no_grad():
            stock_processed = stock.clone()
            stock_processed[stock_processed == -2] = 0.0
            stock_processed[stock_processed == -1] = 1.0
            stock_processed[stock_processed >= 0] = -1.0
        return stock_processed 

def select_action(state, policy):
    # Get action components from policy network
    stock_logits, product_logits, position_mean = policy(state)
    
    # Sample stock index
    stock_dist = Categorical(logits=stock_logits)
    stock_idx = stock_dist.sample()
    stock_log_prob = stock_dist.log_prob(stock_idx)
    
    # Sample product index
    # Mask unavailable products (quantity = 0)
    mask = torch.tensor([p['quantity'] > 0 for p in state['products']], dtype=torch.float32)
    masked_logits = product_logits + (mask + 1e-10).log()
    product_dist = Categorical(logits=masked_logits)
    product_idx = product_dist.sample()
    product_log_prob = product_dist.log_prob(product_idx)
    
    # Get size from selected product
    size = state['products'][product_idx]['size'].tolist()
    
    # Sample position
    position_std = torch.exp(policy.position_log_std)
    position_dist = Normal(position_mean, position_std)
    position_sample = position_dist.sample()
    position_log_prob = position_dist.log_prob(position_sample).sum()
    
    # Total log probability
    total_log_prob = stock_log_prob + product_log_prob + position_log_prob
    policy.saved_log_probs.append(total_log_prob)
    
    # Convert position to valid coordinates
    position_np = position_sample.detach().numpy().squeeze()
    position = [
        int(np.clip(position_np[0].item(), 0, policy.max_width - size[0])),
        int(np.clip(position_np[1].item(), 0, policy.max_height - size[1]))
    ]
    
    action = {
        'stock_idx': stock_idx.item(),
        'size': size,
        'position': tuple(position)
    }
    
    return action

# def finish_episode(policy, optimizer, gamma):
#     R = 0
#     policy_loss = []
#     returns = []
    
#     # Discounted reward calculation
#     for r in policy.rewards[::-1]:
#         R = r + gamma * R
#         returns.insert(0, R)
#     returns = torch.tensor(returns)
#     returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
#     for log_prob, R in zip(policy.saved_log_probs, returns):
#         policy_loss.append(-log_prob * R)
    
#     optimizer.zero_grad()
#     policy_loss = torch.cat(policy_loss).sum()
#     policy_loss.backward()
#     optimizer.step()
    
#     # Clear episode data
#     del policy.rewards[:]
#     del policy.saved_log_probs[:]
#     gc.collect()


def finish_episode(policy, optimizer, gamma):
    if len(policy.rewards) == 0:
        # No valid actions in episode
        policy.saved_log_probs.clear()
        return
        
    R = 0
    policy_loss = []
    returns = []
    
    # Calculate the discounted returns
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    if len(returns) > 1:  # Normalize if more than one return
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # Calculate policy loss
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    if policy_loss:  # Only backpropagate if there are valid actions
        optimizer.zero_grad()
        policy_loss = sum(policy_loss)  # Sum the scalar tensors directly
        policy_loss.backward()
        optimizer.step()
    
    # Clear episode data
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    gc.collect()


# def main():
#     gamma = 0.99
#     seed = 42
#     log_interval = 10
#     num_episodes = 50
    
    
#     env = gym.make(
#         "gym_cutting_stock/CuttingStock-v0",
#         render_mode="human",  # Comment this line to disable rendering    
#     )
#     observation, info = env.reset(seed=seed)
#     torch.manual_seed(seed)
    
#     num_stocks = len(observation["stocks"])
#     max_width, max_height = observation["stocks"][0].shape
#     num_products = len(observation["products"])
    
    
#     policy = Policy(num_stocks, max_width, max_height, num_products)
#     optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    
#     max_invalid_actions = 5
#     invalid_action_count = 0
#     for episode in range(num_episodes):
#         print(f'Episode {episode}')
#         invalid_action_count = 0
#         # step = 0
#         observation, info = env.reset(seed=seed)
#         ep_reward = 0
#         terminated = truncated = False

#         for step in range(600):
#             # print(f'Step {step}')
#             action = select_action(observation, policy)
#             observation, reward, terminated, truncated, info = env.step(action)  # Fixed line
#             print(f'Reward: {reward}')
#             policy.rewards.append(reward)
#             ep_reward += reward
            
#             if reward == -1:
#                 invalid_action_count += 1
#             else:
#                 invalid_action_count = 0
            
#             if invalid_action_count >= max_invalid_actions:
#                 print('Too many invalid actions. Terminating episode.')
#                 terminated = True
            
#             if terminated or truncated:
#                 break

#         finish_episode(policy, optimizer, gamma)

#         if episode % log_interval == 0:
#             print(f'Episode {episode}\tReward: {ep_reward}')


# In main(), optimize training loop
def main():
    gamma = 0.99
    seed = 42
    log_interval = 5
    num_episodes = 1000
    max_steps = 500  # Add maximum steps per episode
    
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",  # Comment this line to disable rendering
    )
    
    observation, info = env.reset(seed=seed)
    torch.manual_seed(seed)
    
    num_stocks = len(observation["stocks"])
    max_width, max_height = observation["stocks"][0].shape
    num_products = len(observation["products"])
    
    policy = Policy(num_stocks, max_width, max_height, num_products)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
    
    best_reward = float('-inf')
    patience = 1000
    no_improve = 0
    
    for episode in range(num_episodes):
        observation, info = env.reset(seed=seed)
        ep_reward = 0
        invalid_count = 0
        
        for step in range(max_steps):
            action = select_action(observation, policy)
            observation, reward, terminated, truncated, info = env.step(action)
            
            policy.rewards.append(reward)
            ep_reward += reward
            
            if reward < 0:
                invalid_count += 1
            else:
                invalid_count = 0
                
            if invalid_count >= 3 or terminated or truncated:
                break
        
        finish_episode(policy, optimizer, gamma)
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break
        
        if episode % log_interval == 0:
            print(f'Episode {episode}\tReward: {ep_reward:.2f}')


if __name__ == '__main__':
    main()