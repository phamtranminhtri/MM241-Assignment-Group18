from policy import Policy
import numpy as np
import scipy
from scipy.signal import correlate

def Policy2310038(Policy):
    from policy import Policy

class Policy2210xxx(Policy):
    called = 0
    new = 1
    stock = []
    sorted_stocks_area = []
    sorted_stocks_indices =[]
    
    def _init_(self):
        pass

    def sort_products(self, prods):
        areas = np.array([prod["size"][0] * prod["size"][1] for prod in prods])
        #sorted_stock = sorted(stocks, key=lambda arr: len(arr)*len(arr[0]),reverse=True)
        sorted_prods_indices = np.argsort(areas)[::-1]
        sorted_prods = []
        for i in sorted_prods_indices:
            sorted_prods.append(prods[i])
        return sorted_prods
    def sort_stocks(self, stocks):
        areas = np.array([np.sum(np.any(stock != -2, axis=1)) * np.sum(np.any(stock != -2, axis=0)) for stock in stocks])
        #sorted_stock = sorted(stocks, key=lambda arr: len(arr)*len(arr[0]),reverse=True)
        sorted_stocks_indices = np.argsort(areas)[::-1]
        sorted_stocks_area = []
        for i in sorted_stocks_indices:
            sorted_stocks_area.append(areas[i])
        return sorted_stocks_area, sorted_stocks_indices
    
    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

    def get_action(self, observation, info):
        prods = observation["products"]
        sorted_prods = self.sort_products(prods)

        if self.called == 0:
            self.stocks = observation["stocks"]
            self.sorted_stocks_area, self.sorted_stocks_indices = self.sort_stocks(self.stocks)
            if np.size(sorted_prods)<=5:
                self.sorted_stocks_indices = np.flip(self.sorted_stocks_indices)
            #print(self.sorted_stocks_area)
            self.called = 1

        for i in self.sorted_stocks_indices:
            for prod in sorted_prods:
                if prod["quantity"]>0:
                    prod_size = np.array(prod["size"])
                    #prod_w, prod_h = prod_size
                    stock = self.stocks[i]
                    spaces = self.find_place(stock, prod_size)   
                    if spaces.size > 0:
                        x, y  = spaces[0]
                        return {
                            "stock_idx": i,
                            "size": prod_size,
                            "position": (x,y),
                        }
                    else:
                        spaces = self.find_place(stock, np.flip(prod_size))
                        if spaces.size > 0:
                            x, y  = spaces[0]
                            return {
                                "stock_idx": i,
                                "size": np.flip(prod_size),
                                "position": (x,y),
                            }
            #self.sorted_stocks_area = np.delete(self.sorted_stocks_area,0,0)
            #self.sorted_stocks_indices = np.delete(self.sorted_stocks_indices,0,0)
            self.new = 1

            
    def find_place(self, stock, prod_size):
        r_rows, r_cols = prod_size
        
        # Find the first rows and columns where there's a valid empty space (-2)
        rows = np.argmin(stock[:, 0] == -2)  # Find the first empty row
        cols = np.argmin(stock[0, :] == -2)  # Find the first empty column
        
        # If there are no empty spaces, adjust the index
        if stock[rows, 0] != -2: 
            rows = stock.shape[0]
        if stock[0, cols] != -2:
            cols = stock.shape[1]
        
        # Pad the stock grid with 1s (empty) and 0s (filled)
        stocker = np.pad(1 - (stock[:rows, :cols] == -1), 1, mode="constant", constant_values=1)

        # Define convolution kernels
        rect_kernel = np.ones((r_rows, r_cols), dtype=int)
        up_kernel = np.ones((1, r_cols), dtype=int)
        left_kernel = np.ones((r_rows, 1), dtype=int)

        # Check for valid placements using convolution
        result1 = correlate(stocker, rect_kernel)[r_rows:-1, r_cols:-1] == 0  # Check rectangular area
        result2 = correlate(stocker, up_kernel)[0:-2, r_cols:-1] >= 1         # Check upward continuity
        result3 = correlate(stocker, left_kernel)[r_rows:-1, 0:-2] >= 1       # Check left continuity
        
        # Combine the results to find all valid placement locations
        result = result1 & result2 & result3
        
        # Get the positions (corners) where the product can fit
        corners = np.argwhere(result)
        return corners