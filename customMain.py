import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
import pygame
pygame.display.set_mode((800, 800))
from customEnv import CustomCuttingStockEnv, read_testcase_file
# Create the environment
# env = gym.make(
#     "gym_cutting_stock/CuttingStock-v0",
#     render_mode="human",  # Comment this line to disable rendering
# )
env = CustomCuttingStockEnv(render_mode="human")
data = read_testcase_file('data.txt')

NUM_EPISODES = 100
if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed=42, options=data)

    # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep,options=data)
            print(info)
            ep += 1

    # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1

    # Uncomment the following code to test your policy
    # # Reset the environment
    # observation, info = env.reset(seed=42, options=data)
    # print(info)
    # ep = 0
    # policy2210xxx = Policy2210xxx()
    # while ep < NUM_EPISODES:
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
        
    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(options=data)
    #         ep += 1

env.close()
