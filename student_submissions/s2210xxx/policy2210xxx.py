from policy import Policy
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.stock_idx = 0
            self.corner_points = {0:[(0,0)]}
        elif policy_id == 2:
            pass    

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            if not info["filled_ratio"]:
                self.stock_idx = 0
                self.corner_points.clear()
            list_prods = sorted(observation["products"], key=lambda prod: prod["size"][0], reverse=True)
            if self.stock_idx not in self.corner_points:
                self.corner_points[self.stock_idx] = [(0,0)]
            stock_w,stock_h = self._get_stock_size_(observation["stocks"][self.stock_idx])
            # Find first prod["quantity"] > 0
            #prod = next((prod for prod in list_prods if prod["quantity"] > 0), None)
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w,prod_h = prod_size
                    for x,y in self.corner_points[self.stock_idx]:
                        if (self._can_place_(observation["stocks"][self.stock_idx], (x, y), (prod_w, prod_h)) and
                            x + prod_w <= stock_w and y + prod_h <= stock_h):
                            action ={
                                "stock_idx": self.stock_idx,
                                "size": (prod_w,prod_h),
                                "position":(x,y),
                            }
                            # Cập nhật điểm góc
                            self.corner_points[self.stock_idx].append((x + prod_w, y))
                            self.corner_points[self.stock_idx].append((x, y + prod_h))
                            self.corner_points[self.stock_idx].remove((x, y))
                            return action                            
            # Không còn chỗ đặt
            self.stock_idx += 1
            return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}
            
