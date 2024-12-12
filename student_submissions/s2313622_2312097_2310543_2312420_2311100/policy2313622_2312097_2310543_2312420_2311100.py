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
            self.policy = self.HeuristicPolicy()
        
        elif policy_id == 2:
            # ------------------------------------------------------------------------------#
            # NOTE: The following policy was trained for the default gym_cutting_stock      #
            # environment parameter:                                                        #
            # min_w=50,                                                                     #
            # min_h=50,                                                                     #
            # max_w=100,                                                                    #
            # max_h=100,                                                                    #
            # num_stocks=100,                                                               #
            # max_product_type=25,                                                          #
            # max_product_per_type=20,                                                      #
            # If you use different parameters, you have to retrain the policy for your      #
            # environment, using the provided ppo_optimized.py script; otherwise, the       #
            # policy will not work correctly.                                               #
            # ------------------------------------------------------------------------------#
            self.num_stocks = 100
            self.num_products = 25

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            return self.policy.get_action(observation, info)
        
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
    
    class HeuristicPolicy(Policy):
        def __init__(self):
            self.prod_indice = []
            self.sorted_stock = []
            self.solutions = []
            self.actions = []

        def get_action(self, observation, info):
        
            if info["filled_ratio"] == 0:
                self.reset(observation["products"], observation["stocks"])

            while self.actions or self.solutions or self.sorted_stock:

                if not self.actions:
                    self.new_action(observation["products"])

                if self.actions:
                    return self.actions.pop(0)
            
                if not self.solutions:
                    self.new_solution(observation["products"])

                if not self.solutions:
                    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        def reset(self, prods, stocks):
            self.actions = []
            self.solutions = []
            self.prod_indice = sorted(range(len(prods)), key=lambda i: prods[i]["size"][0] * prods[i]["size"][1], reverse=True)
            self.sorted_stock = sorted([(self._get_stock_size_(stock), i) for i, stock in enumerate(stocks)], key=lambda x: x[0][0] * x[0][1], reverse=False)

        def new_solution(self, prods):
            if not self.prod_indice:
                return

            while prods[self.prod_indice[0]]["quantity"] == 0:
                self.prod_indice.pop(0)

            prod_w, prod_h = prods[self.prod_indice[0]]["size"]
            quantity = prods[self.prod_indice[0]]["quantity"]

            # find smallest prod can be used
            max_stock = 0
            _max = -1

            for stock in self.sorted_stock:
                stock_w, stock_h = stock[0]
                if stock_w >= prod_w and stock_h >= prod_h:
                    if (stock_w // prod_w) * (stock_h // prod_h) > _max:
                        _max = (stock_w // prod_w) * (stock_h // prod_h)
                        max_stock = stock
                        if _max >= quantity:
                            break

                if stock_w >= prod_h and stock_h >= prod_w:
                    if (stock_w // prod_h) * (stock_h // prod_w) > _max:
                        _max = (stock_w // prod_h) * (stock_h // prod_w)
                        max_stock = stock
                        if _max >= quantity:
                            break

            if _max == -1:
                return
            self.solutions = [{"stock_idx": max_stock[1], "size": max_stock[0], "position": (0, 0)}]
            self.sorted_stock.remove(max_stock)

        def new_action(self, prods):
            # choose product
            if not self.solutions:
                return

            stock_idx = self.solutions[0]["stock_idx"]

            while self.solutions:
                solution = self.solutions[0]
                stock_w, stock_h = solution["size"]
                self.solutions.pop(0)

                # Try to find the first fit product
                for prod_idx in self.prod_indice:
                    prod = prods[prod_idx]
            
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        prod_w, prod_h = prod_size

                        if stock_w >= prod_w and stock_h >= prod_h:
                            if stock_w >= prod_h and stock_h >= prod_w:
                                if (stock_w // prod_w) * (stock_h // prod_h) >= (stock_w // prod_h) * (stock_h // prod_w):
                                    self.get_solution(solution, prod)
                                    return
                                else:
                                    prod["size"] = prod["size"][::-1]
                                    self.get_solution(solution, prod)
                                    return
                            self.get_solution(solution, prod)
                            return
                        else:
                            if stock_w >= prod_h and stock_h >= prod_w:
                                prod["size"] = prod["size"][::-1]
                                self.get_solution(solution, prod)
                                return

        def get_solution(self, solution, prod):
            quantity = prod["quantity"]
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            stock_w, stock_h = solution["size"]
            stock_x, stock_y = solution["position"]
            stock_idx = solution["stock_idx"]

            cut_w = stock_w // prod_w
            cut_h = stock_h // prod_h

            if cut_w * cut_h > quantity:
                cut_h = quantity // cut_w
            if cut_h == 0:
                cut_w = quantity
                cut_h = 1

            for x in range(cut_w):
                for y in range(cut_h):
                    self.actions.append({"stock_idx": stock_idx, "size": prod_size, "position": (stock_x + x * prod_w, stock_y + y * prod_h)})

            prod_w *= cut_w
            prod_h *= cut_h

            # add solutions
            if (stock_w - prod_w) * stock_h < (stock_h - prod_h) * stock_w:
                if stock_w - prod_w != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [stock_w - prod_w, prod_h], "position": (stock_x + prod_w, stock_y)})
                if stock_h - prod_h != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [stock_w, stock_h - prod_h], "position": (stock_x, stock_y + prod_h)})
            else:
                if stock_w - prod_w != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [stock_w - prod_w, stock_h], "position": (stock_x + prod_w, stock_y)})
                if stock_h - prod_h != 0:
                    self.solutions.append({"stock_idx": stock_idx, "size": [prod_w, stock_h - prod_h], "position": (stock_x, stock_y + prod_h)})

            self.solutions = sorted(self.solutions, key=lambda x: min(x["size"]), reverse=False)