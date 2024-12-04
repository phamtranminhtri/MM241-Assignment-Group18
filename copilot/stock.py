import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class CuttingStockAgent:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state.flatten())
        probs = self.policy_network(state)
        action = np.random.choice(len(probs.detach().numpy()), p=probs.detach().numpy())
        return action

    def store_transition(self, state, action, reward):
        self.episode_states.append(state.flatten())
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def compute_returns(self, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def update_policy(self):
        returns = self.compute_returns()
        states = torch.FloatTensor(self.episode_states)
        actions = torch.LongTensor(self.episode_actions)

        self.optimizer.zero_grad()
        log_probs = torch.log(self.policy_network(states))
        selected_log_probs = returns * log_probs[np.arange(len(actions)), actions]
        loss = -selected_log_probs.mean()
        loss.backward()
        self.optimizer.step()

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

print("Stock agent loaded")

# Example usage
state_dim = 100  # Flattened grid size
action_dim = 100  # Number of stocks * number of products * grid positions
agent = CuttingStockAgent(state_dim, action_dim)

# Simulate an episode
state = np.random.rand(100, 100)  # Example state
for _ in range(10):  # Example episode length
    action = agent.select_action(state)
    next_state = np.random.rand(100, 100)  # Example next state
    reward = np.random.rand()  # Example reward
    agent.store_transition(state, action, reward)
    state = next_state

agent.update_policy()