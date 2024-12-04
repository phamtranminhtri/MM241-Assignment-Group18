import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = self.softmax(self.fc2(x))
        return action_probs

# Define the REINFORCE algorithm
class REINFORCE:
    def __init__(self, policy, lr=1e-3, gamma=0.99):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma

    def compute_returns(self, rewards):
        """Compute the discounted cumulative rewards."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def update_policy(self, log_probs, returns):
        """Update the policy using policy gradients."""
        loss = -torch.sum(torch.stack(log_probs) * returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Main loop
if __name__ == "__main__":
    env = gym.make(
        "CartPole-v1",
        # render_mode="human"
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    reinforce = REINFORCE(policy)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = policy(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            log_probs.append(action_dist.log_prob(action))
            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)

        returns = reinforce.compute_returns(rewards)
        reinforce.update_policy(log_probs, returns)

        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward: {sum(rewards)}")

    env.close()
