from policy import Policy
from .ppo_optimized import PPO, ActorNetwork, CriticNetwork
import torch
import numpy as np
from torch.distributions import Categorical
import os


class Policy2313622_2312097_2310543_2312420_2311100(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.id = policy_id
            pass
        
        elif policy_id == 2:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_stocks = 100
            self.num_products = 25
            policy = ActorNetwork(num_stocks=self.num_stocks, num_products=self.num_products, device=device).to(device)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            policy_path = os.path.join(current_dir, "ppo_actor.pth")
            policy.load_state_dict(
                torch.load(
                    policy_path,
                    map_location=device
                )
            )
            self.id = policy_id
            self.policy = policy
            

    def get_action(self, observation, info):
        # Student code here
        if self.id == 1:
            pass
        
        elif self.id == 2:
            obs = observation
            
            # Extract observation components
            stocks_np = obs['stocks']  # shape (num_stocks, 100, 100)
            products_np = obs['products']  # shape (num_products, 3)
            
            # Extract numerical data from products_np
            # Extract product features and quantities
            products_list = []
            for product in products_np:
                size = product['size']
                quantity = product['quantity']
                product_features = np.concatenate((size, [quantity]))
                products_list.append(product_features)

            # Calculate padding length
            pad_length = self.num_products - len(products_list)

            # Pad both arrays if needed
            if pad_length > 0:
                products_list += [[0, 0, 0]] * pad_length

            # Convert to numpy arrays
            products_array = np.array(products_list)  # Shape: (num_products, 3)
            
            # Query the actor network for a mean action
            stock_logits, product_logits = self.policy(obs)
            
            # Mask out products whose quantity is 0
            for i, product in enumerate(products_array):
                if product[2] == 0:
                    product_logits[0][i] = -float('inf')

            # Sample an action from the distribution
            product_dist = Categorical(logits=product_logits)
            product_action = product_dist.sample()
            products_size = [products_array[product_action.item()][0], products_array[product_action.item()][1]]

            # Mask out stocks where product won't fit
            for i, stock in enumerate(stocks_np):
                act = self.greedy(obs['stocks'], i, products_size)
                if act['stock_idx'] == -1:
                    stock_logits[0][i] = -float('inf')

            stock_dist = Categorical(logits=stock_logits)
            stock_action = stock_dist.sample()
            
            # Move action results back to CPU for numpy operations
            stock_action = stock_action.cpu()
            product_action = product_action.cpu()

            # Product size [w, h]
            action = self.greedy(obs['stocks'], stock_action.item(), products_size)

            # Return the sampled action and the log probability of that action in our distribution
            return action

    # Student code here
    # You can add more functions if needed
    def greedy(self, stocks, stock_idx, prod_size):
        if prod_size == [0, 0]:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
        stock = stocks[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w >= prod_w and stock_h >= prod_h:
            for x in range(stock_w - prod_w + 1):
                find = 0
                for y in range(stock_h - prod_h + 1):
                    if stock[x][y] == -1:
                        if find == 0:
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}
                            find = 1
                    else:
                        if find == 1:
                            find = 0

        if stock_w >= prod_h and stock_h >= prod_w:
            for x in range(stock_w - prod_h + 1):
                find = 0
                for y in range(stock_h - prod_w + 1):
                    if stock[x][y] == -1:
                        if find == 0:
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (x, y)}
                            find = 1
                    else:
                        if find == 1:
                            find = 0

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}