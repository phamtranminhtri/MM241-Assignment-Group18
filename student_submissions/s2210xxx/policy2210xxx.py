from policy import Policy
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.corner_points = {0: [(0,0)]}
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            list_prods = sorted(observation["products"], key=lambda prod: prod["size"][0], reverse=True)
            if not info["filled_ratio"]:
                self.corner_points = {0: [(0,0)]}
            # Find first prod["quantity"] > 0
            prod = next((prod for prod in list_prods if prod["quantity"] > 0), None)
            if not prod:
                return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}
            
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                if stock_idx not in self.corner_points:
                    self.corner_points[stock_idx] = [(0, 0)]
                for x, y in self.corner_points[stock_idx]:
                    if (self._can_place_(observation["stocks"][stock_idx], (x, y), (prod_w, prod_h)) and
                        x + prod_w <= stock_w and y + prod_h <= stock_h):
                        
                        action = {
                            "stock_idx": stock_idx,
                            "size": (prod_w, prod_h),
                            "position": (x, y),
                        }
                        # Cập nhật điểm góc
                        self.corner_points[stock_idx].append((x + prod_w, y))
                        self.corner_points[stock_idx].append((x, y + prod_h))
                        self.corner_points[stock_idx].remove((x, y))

                        # Thêm tấm vào used_stocks
                        return action

        pass

    # Student code here
    # You can add more functions if needed
