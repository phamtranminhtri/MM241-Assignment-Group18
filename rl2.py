import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import gym_cutting_stock
import numpy as np
import gc

class Policy(nn.Module):
    def __init__(self, num_stocks, max_width, max_height, num_products):
        super(Policy, self).__init__()
        
        self.num_stocks = num_stocks
        self.max_width = max_width
        self.max_height = max_height
        self.num_products = num_products
        
        # Larger network for better feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Corrected calculation
        h_out = max_height // 4
        w_out = max_width // 4
        self.conv_output_size = 32 * h_out * w_out  # Should now match the tensor dimensions
        
        # Separate branches for stocks and products
        self.stock_fc = nn.Linear(self.conv_output_size, 64)
        self.product_fc = nn.Linear(num_products * 3, 64)
        
        # Combine features
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Output heads
        self.stock_head = nn.Linear(32, num_stocks)
        self.product_head = nn.Linear(32, num_products)
        self.position_head = nn.Linear(32, 2)
        
        # Learnable but constrained position std
        self.position_log_std = nn.Parameter(torch.zeros(2))
        
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, state):
        # Process stocks with attention to available space
        stocks_features = []
        for stock in state['stocks']:
            stock_tensor = self.preprocess_stock(torch.tensor(stock, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
            x = F.relu(self.conv1(stock_tensor))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            stocks_features.append(x.view(-1))
        
        x = torch.stack(stocks_features).mean(dim=0)
        x = F.relu(self.stock_fc(x))
        
        # Process products with attention to quantities
        products_info = np.array([
            np.concatenate([
                p['size'].astype(np.float32) / np.array([self.max_width, self.max_height]),
                [p['quantity'] / self.num_products]
            ])
            for p in state['products']
        ])
        products_tensor = torch.from_numpy(products_info).float().view(-1)
        
        products_tensor = torch.FloatTensor(products_info).view(-1)
        p = F.relu(self.product_fc(products_tensor))
        
        # Combine features
        combined = torch.cat([x, p])
        hidden = F.relu(self.fc1(combined))
        hidden = F.relu(self.fc2(hidden))
        
        # Output with constraints
        stock_logits = self.stock_head(hidden)
        product_logits = self.product_head(hidden)
        position_mean = torch.sigmoid(self.position_head(hidden)) * torch.tensor([self.max_width, self.max_height])
        
        return stock_logits, product_logits, position_mean
            
    def preprocess_stock(self, stock):
        with torch.no_grad():
            stock_processed = stock.clone()
            stock_processed[stock_processed == -2] = 0.0
            stock_processed[stock_processed == -1] = 1.0
            stock_processed[stock_processed >= 0] = -1.0
        return stock_processed 

def select_action(state, policy):
    stock_logits, product_logits, position_mean = policy(state)
    
    # Stock selection
    stock_mask = torch.tensor([1.0 if np.any(s == -1) else 0.0 for s in state['stocks']])
    if stock_mask.sum() == 0:
        # No available stocks
        return None
    stock_probs = F.softmax(stock_logits + (stock_mask + 1e-10).log(), dim=0)
    stock_dist = Categorical(probs=stock_probs)
    stock_idx = stock_dist.sample()
    stock_log_prob = stock_dist.log_prob(stock_idx)
    
    # Product selection
    product_mask = torch.tensor([1.0 if p['quantity'] > 0 else 0.0 for p in state['products']])
    if product_mask.sum() == 0:
        # No available products
        return None
    product_probs = F.softmax(product_logits + (product_mask + 1e-10).log(), dim=0)
    product_dist = Categorical(probs=product_probs)
    product_idx = product_dist.sample()
    product_log_prob = product_dist.log_prob(product_idx)
    
    # Get product size
    size = state['products'][product_idx]['size']
    
    # Position selection
    position_std = torch.exp(policy.position_log_std).clamp(0.1, 1.0)
    position_dist = Normal(position_mean, position_std)
    position_sample = position_dist.sample()
    position_np = position_sample.detach().numpy()
    position = [
        int(np.clip(position_np[0], 0, policy.max_width - size[0])),
        int(np.clip(position_np[1], 0, policy.max_height - size[1]))
    ]
    
    # Log probability of position
    position_log_prob = position_dist.log_prob(torch.tensor(position)).sum()
    total_log_prob = stock_log_prob + product_log_prob + position_log_prob
    policy.saved_log_probs.append(total_log_prob)
    
    return {
        'stock_idx': stock_idx.item(),
        'size': size,
        'position': tuple(position)
    }

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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
    
    # Clear episode data
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    gc.collect()

def main():
    gamma = 0.99
    seed = 42
    log_interval = 5
    num_episodes = 1000
    max_steps = 200
    
    # Initialize environment with smaller settings for faster training
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",          # Disable rendering for faster training if needed
        max_w=100,                     # Start with smaller size
        max_h=100,
        num_stocks=25,                 # Fewer stocks initially
        max_product_type=3,           # Fewer product types
    )
    
    observation, info = env.reset(seed=seed)
    torch.manual_seed(seed)
    
    num_stocks = len(observation["stocks"])
    max_width, max_height = observation["stocks"][0].shape
    num_products = len(observation["products"])
    
    policy = Policy(num_stocks, max_width, max_height, num_products)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    best_reward = float('-inf')
    patience = 1000
    no_improve = 0
    
    for episode in range(num_episodes):
        observation, info = env.reset(seed=seed)
        ep_reward = 0
        invalid_count = 0
        
        # Anneal exploration
        exploration_factor = max(0.1, 1.0 - episode / 200)  # Decay over 200 episodes
        policy.position_log_std.data = torch.ones(2) * exploration_factor
        
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
            print(f'Episode {episode}\tReward: {ep_reward:.2f}\tInvalid: {invalid_count}')

if __name__ == '__main__':
    main()