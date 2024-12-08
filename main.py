import gym_cutting_stock
import gymnasium as gym
import numpy as np
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

import pygame
pygame.display.set_mode((800, 800))

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)

NUM_EPISODES = 5

if __name__ == "__main__":
    # Test Policy 1
    # Reset the environment
    observation, info = env.reset(seed=42)

    ep = 0
    gd_policy = Policy2210xxx(policy_id=1)
    while ep < NUM_EPISODES:
        
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            ep += 1
    # Test RandomPolicy
    # Reset the environment
    observation, info = env.reset(seed=42)

    rd_policy = RandomPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = rd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
           
            ep += 1

env.close()
