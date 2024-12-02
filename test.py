import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Function to compute discounted rewards
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

# REINFORCE Implementation
def reinforce(env_name="CartPole-v1", episodes=1000, gamma=0.99, lr=0.01, render=False):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Convert state properly to a PyTorch tensor
            state_tensor = torch.FloatTensor(np.array(state))
            
            action_probs = policy(state_tensor)
            action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
            
            log_prob = torch.log(action_probs[action])
            next_state, reward, done, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            
            # Debugging state if issues persist
            if len(next_state) != input_dim:
                raise ValueError(f"Unexpected state shape: {next_state}")
            
            state = next_state
        
        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Normalize rewards for stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Compute loss
        loss = 0
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss += -log_prob * reward
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        if total_reward >= 500:
            print(f"Solved after {episode + 1} episodes!")
            break
    
    env.close()


# Train the agent
reinforce(episodes=1000, gamma=0.99, lr=0.01, render=False)
