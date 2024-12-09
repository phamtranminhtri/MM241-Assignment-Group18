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
            self.active_stocks = []  # List of used stocks
            self.unused_stocks = []  # List of unused stock    

    def get_action(self, observation, info):
        # Student code here
        # Policy 1
        if self.policy_id == 1:
            # Reset all attributes in new episode
            if not info["filled_ratio"]:
                self.stock_idx = 0
                self.corner_points.clear()
                # Sort the list of products 
                self.list_prods = sorted(observation["products"], key=lambda prod: prod["size"][0], reverse=True)

            # Initialize first corner point if current stock is not placed
            if self.stock_idx not in self.corner_points:
                self.corner_points[self.stock_idx] = [(0,0)]
            stock_w,stock_h = self._get_stock_size_(observation["stocks"][self.stock_idx])
            # Loop through all products, place all possible to current stock
            for prod in self.list_prods:
                # Pick a product
                if prod["quantity"] <=0:
                    continue

                prod_w, prod_h = prod["size"]
                # Find a corner point to place this product
                for x,y in self.corner_points[self.stock_idx]:
                    if stock_w < x + prod_w or stock_h <y + prod_h: # Product can't fit current stock
                        continue
                    # Check if this product doesnt overlap the others
                    if self._can_place_(observation["stocks"][self.stock_idx], (x, y), (prod_w, prod_h)):
                        action ={
                            "stock_idx": self.stock_idx,
                            "size": (prod_w,prod_h),
                            "position":(x,y),
                        }
                        # Update corner points list
                        self.corner_points[self.stock_idx].append((x + prod_w, y))
                        self.corner_points[self.stock_idx].append((x, y + prod_h))
                        self.corner_points[self.stock_idx].remove((x, y))
                        return action
            # Move to next stock if cant place any product
            self.stock_idx += 1
            return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}
        
        # Policy 2
        if self.policy_id == 2:
            # Reset when enviroment reset
            if not info["filled_ratio"]:
                self.active_stocks = []
                self.unused_stocks = []

            if not self.unused_stocks:
                self.sorted_products = sorted(
                    observation['products'], 
                    key=lambda x: x['size'][0] * x['size'][1], 
                    reverse=True
                )
                self.unused_stocks= sorted(
                 [{'index': i, 'stock': stock} for i, stock in enumerate(observation['stocks'])],
                key=lambda x: self._get_stock_size_(x['stock'])[0] * self._get_stock_size_(x['stock'])[1])

            for product in self.sorted_products:
                if product['quantity'] > 0:
                    prod_size = product['size']
                    best_fit_stock_idx = None
                    best_fit_position = None
                    min_waste = float('inf')

                    # Scan stock in use to find stock with the least wasted space
                    for stock_data in self.active_stocks:
                        stock_idx = stock_data['index']

                        stock = stock_data['stock']
                        position = self.find_best_position(stock, prod_size[0], prod_size[1])

                        if position:
                            waste = self.calculate_waste(stock, position[0], position[1], prod_size[0], prod_size[1])
                            if waste < min_waste:
                                min_waste = waste
                                best_fit_stock_idx = stock_idx
                                best_fit_position = position

                    # If not found in active_stocks, select smallest stock from unused_stocks
                    if best_fit_stock_idx is None and self.unused_stocks:

                        smallest_stock_data = self.unused_stocks.pop(0)  # Lấy stock nhỏ nhất
                        self.active_stocks.append(smallest_stock_data)  # Đưa vào active_stocks
                        stock_idx = smallest_stock_data['index']
                        stock = smallest_stock_data['stock']
                        position = self.find_best_position(stock, prod_size[0], prod_size[1])
                        if position:
                            best_fit_stock_idx = stock_idx
                            best_fit_position = position

                    # Place the product if you find a suitable location
                    if best_fit_stock_idx is not None:
                        return {
                            'stock_idx': best_fit_stock_idx,
                            "size": (prod_size[0],prod_size[1]),
                            'position': best_fit_position
                        }
            # Out of product
            return {'stock_idx': -1, 'size': (0, 0), 'position': (0, 0)}
    
    def find_best_position(self, stock, width, height):
        best_pos = None
        min_waste = float('inf')
        rows, cols = self._get_stock_size_(stock)

        for i in range(rows - width + 1):
            for j in range(cols - height + 1):
                if self._can_place_(stock,(i, j), (width, height)) and i + width <= rows and j + height <= cols:

                    waste = self.calculate_waste(stock, i, j, width, height)
                    if waste < min_waste:
                        min_waste = waste
                        best_pos = (i, j)
        return best_pos            

    def calculate_waste(self, stock, x, y, width, height):
        empty_area = np.count_nonzero(stock[x:x+width, y:y+height] == -1)
        product_area = width * height
        return empty_area - product_area