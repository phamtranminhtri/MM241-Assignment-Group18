from policy import Policy
import numpy as np
import gym_cutting_stock
import gymnasium as gym
class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.stock_idx = 0
            self.corner_points = {0:[(0,0)]}
        elif policy_id == 2:
            self.active_stocks = []  # Danh sách các stock đang được sử dụng
            self.unused_stocks = []  # Danh sách các stock chưa được sử dụng
            pass    

    def get_action(self, observation, info):
        # Student code here
        #print(info)
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
                        if (self._can_place_(observation["stocks"][self.stock_idx], (x, y), (prod_w, prod_h)) ):
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
        if self.policy_id == 2:
            
            max_colors = 10
            
            # if not info["filled_ratio"]:
            #     print("bbbbb")
            #     self.active_stocks.clear()  # Danh sách các stock đang được sử dụng
            #     self.unused_stocks.clear() # Danh sách các stock chưa được sử dụng
            if not self.unused_stocks:
                self.unused_stocks= sorted(
                 [{'index': i, 'stock': stock} for i, stock in enumerate(observation['stocks'])],
                key=lambda x: self._get_stock_size_(x['stock'])[0] * self._get_stock_size_(x['stock'])[1])
                
            # Sắp xếp sản phẩm theo diện tích giảm dần một lần duy nhất    
            if not hasattr(self, 'sorted_products'):
                self.sorted_products = sorted(
                    observation['products'], 
                    key=lambda x: x['size'][0] * x['size'][1], 
                    reverse=True
                )
            for product_idx, product in enumerate(self.sorted_products):
                if product['quantity'] > 0:
                    prod_size = product['size']
                    best_fit_stock_idx = None
                    best_fit_position = None
                    min_waste = float('inf')

                    # Quét qua các stock đang được sử dụng để tìm stock có wasted area nhỏ nhất
                    for stock_data in self.active_stocks:
                        stock_idx = stock_data['index']
                       # print(f"index {stock_idx}")
                        stock = stock_data['stock']
                        position = self.find_best_position(stock, prod_size[0], prod_size[1])

                        if position:
                            waste = self.calculate_waste(stock, position[0], position[1], prod_size[0], prod_size[1])
                            if waste < min_waste:
                                min_waste = waste
                                best_fit_stock_idx = stock_idx
                                best_fit_position = position
                               # print(best_fit_stock_idx)
                    # Nếu không tìm thấy trong active_stocks, chọn stock nhỏ nhất từ unused_stocks
                    if best_fit_stock_idx is None and self.unused_stocks:
                        smallest_stock_data = self.unused_stocks.pop(0)  # Lấy stock nhỏ nhất
                        self.active_stocks.append(smallest_stock_data)  # Đưa vào active_stocks
                        stock_idx = smallest_stock_data['index']
                        stock = smallest_stock_data['stock']
                        position = self.find_best_position(stock, prod_size[0], prod_size[1])
                        if position:
                            best_fit_stock_idx = stock_idx
                            best_fit_position = position

                    # Đặt sản phẩm nếu tìm được vị trí phù hợp
                    if best_fit_stock_idx is not None:
                        product_id = product_idx % max_colors
                        #self.update_stock(observation['stocks'][best_fit_stock_idx], best_fit_position, prod_size[0], prod_size[1], product_id, max_colors)
                        print(f"size{self._get_stock_size_(observation['stocks'][best_fit_stock_idx]) }")
                        return {
                            'stock_idx': best_fit_stock_idx,
                            #'stock_idx': 71,
                            "size": (prod_size[0],prod_size[1]),
                            'position': best_fit_position
                        }
            return {'stock_idx': -1, 'size': (0, 0), 'position': (0, 0)}
    def find_best_position(self, stock, width, height):
        best_pos = None
        min_waste = float('inf')
        rows, cols = self._get_stock_size_(stock)

        for i in range(rows - height + 1):
            for j in range(cols - width + 1):
                if self._can_place_(stock,(i, j), (width, height)):
                    waste = self.calculate_waste(stock, i, j, width, height)
                    if waste < min_waste:
                        min_waste = waste
                        best_pos = (i, j)
        return best_pos            

    def calculate_waste(self, stock, x, y, width, height):
        empty_area = np.count_nonzero(stock[x:x+height, y:y+width] == -1)
        product_area = width * height
        return empty_area - product_area


    # def update_stock(self, stock, position, width, height, product_id, max_colors):
    #     x, y = position
    #     product_id = product_id % max_colors  # Giới hạn product_id trong phạm vi max_colors
    #     stock[x:x+height, y:y+width] = product_id + 1  # Tránh gán 0, vì 0 thường là màu nền
           
