import gym_cutting_stock
import gymnasium as gym
import numpy as np
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
import time
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 1

if __name__ == "__main__":
    for ep in range(NUM_EPISODES):
        # #Uncomment the following code to test your policy
        # # Reset the environment
        # observation, info = env.reset(seed=42)
        # print(info)
        # policy2210xxx = Policy2210xxx(policy_id=1)
        # stock_idx = 0
        # filled_area = 0
        # total_area = 0    
        # start = time.time()
        # all_pieces_placed = False
        # while not all_pieces_placed:
        #     # Logic để đặt mảnh
        #     action = policy2210xxx.get_action(observation, info, stock_idx)
        #     if action is None:
        #         stock_idx += 1
        #         continue
        #     observation, reward, terminated, truncated, info = env.step(action)
        #     prod_w, prod_h = action["size"]
        #     filled_area += prod_w * prod_h
        #     print(f"filled are = {filled_area}")
        #     # Kiểm tra điều kiện kết thúc khi tất cả mảnh đã được đặt
        #     if all(m["quantity"] <= 0 for m in observation["products"]):
        #         all_pieces_placed = True
        #     # Reset môi trường nếu cần
        #     if terminated or truncated:
        #         observation, info = env.reset(seed=42)  # Reset môi trường với dữ liệu tùy chỉnh

        # # Tính toán diện tích dư chỉ trên các tấm đã được sử dụng
        # for stock_idx in policy2210xxx.used_stocks:
        #     stock = observation["stocks"][stock_idx]
        #     stock_w, stock_h = policy2210xxx._get_stock_size_(observation["stocks"][stock_idx])
        #     print(f"stock_w = {stock_w},stock_h ={stock_h}")
        #     total_area += stock_w * stock_h
        # wasted_area = total_area - filled_area
        # wasted_ratio = (wasted_area / total_area)*100
        # end = time.time()
        # execution = end - start
        # print(f"Episode {ep + 1}: Total_area = {total_area}, Wasted area = {wasted_area} , wasted_ratio = {wasted_ratio} %, Execution time = {execution}")
        # if terminated or truncated:
        #     observation, info = env.reset()
        start_time = time.time()
        filled_area = 0
        total_area = 0
        observation, info = env.reset(seed=42)
        
        policy2210xxx = Policy2210xxx(policy_id=2)
        all_pieces_placed = False
        stock_idx = 0
        
        while not all_pieces_placed:
            action = policy2210xxx.get_action(observation, info, stock_idx)
            if action['stock_idx'] == -1:
                stock_idx += 1
                if stock_idx >= len(observation['stocks']):
                    print("No more available stocks.")
                    break
                continue

            observation, reward, terminated, truncated, info = env.step(action)
            prod_w, prod_h = action["size"]
            
            if action['stock_idx'] != -1:
                filled_area += prod_w * prod_h

            if all(m["quantity"] <= 0 for m in observation["products"]):
                all_pieces_placed = True

            if terminated or truncated:
                break

        total_area = sum(np.prod(observation["stocks"][idx].shape) for idx in policy2210xxx.used_stocks if idx < len(observation["stocks"]))
        wasted_area = total_area - filled_area
        wasted_ratio = (wasted_area / total_area) * 100
        
        end_time = time.time()  # Lấy thời gian kết thúc testcase
        execution_time = end_time - start_time  # Tính thời gian thực hiện

        print(f"Episode {ep + 1}: Total_area = {total_area}, Wasted area = {wasted_area}, Wasted ratio = {round(wasted_ratio, 3)}%")
        print(f"Execution time: {execution_time:.3f} seconds")  # In thời gian thực hiện
env.close()
