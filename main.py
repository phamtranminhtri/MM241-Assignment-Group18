import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx, BFDPolicy, BranchAndBoundPolicy, Policy2312291

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 2

if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    ep = 0
    gd_results = []
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # print(info)
            print(f"Episode {ep}: {info['filled_ratio']}")
            gd_results.append(info)
            observation, info = env.reset(seed=ep)
            ep += 1

    # Reset the environment
    observation, info = env.reset(seed=42)
    
    
    # policy = Policy2210xxx()
    # ep = 0
    # policy_results = []
    # while ep < NUM_EPISODES:
    #     action = policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         policy_results.append(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1
    
    
    policy = BFDPolicy()
    ep = 0
    policy_results = []
    while ep < NUM_EPISODES:
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # print(info)
            print(f"Episode {ep}: {info['filled_ratio']}")
            policy_results.append(info)
            observation, info = env.reset(seed=ep)
            ep += 1
            
    # compare the results
    equal, better = 0, 0
    for i in range(NUM_EPISODES):
        if gd_results[i]["filled_ratio"] == policy_results[i]["filled_ratio"]:
            equal += 1
        elif gd_results[i]["filled_ratio"] > policy_results[i]["filled_ratio"]:
            better += 1
            
    print(f"Equal: {equal}, Better: {better}")
        

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         # observation, info = env.reset(seed=ep)
    #         ep += 1

    # Uncomment the following code to test your policy
    # Reset the environment
    # observation, info = env.reset(seed=42)
    # print(info)

    # policy2210xxx = Policy2210xxx()
    # for _ in range(200):
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(info)

    #     if terminated or truncated:
    #         observation, info = env.reset()

env.close()
