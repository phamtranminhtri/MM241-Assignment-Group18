from policy import Policy
import torch
from torch import nn, optim
import numpy as np
from numpy import random
import os
import copy
from math import exp

class SortedList:
    """
    A list container that maintains ascending order after inserting a new element.
    """
    def __init__(self, sorted_array):
        self.sorted_array = sorted_array
    
    def add(self, new_element):
        for i, element in enumerate(self.sorted_array):
            if element >= new_element:
                self.sorted_array.insert(i, new_element)
                return
            
        self.sorted_array.append(new_element)

    def __getitem__(self, index):
        return self.sorted_array[index]
    
class GreedySAPolicy(Policy):
    """
    This policy first applies a greedy approach, followed by simulated annealing to refine the result. 
    The greedy approach selects the largest remaining product and places it into the largest available stock.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.first_time = True
        self.step = 0
        self.num_stocks = 0
        self.num_products = 0
        self.num_items = 0
        self.stocks = []
        self.stock_indices = None
        self.products = []
        self.product_indices = None
        self.state_matrix = []
        self.actions = []

    @staticmethod
    def place_item(stock, product_width, product_height):
        occupied_cells = stock["occupied_cells"]
        num_row = len(occupied_cells)
        num_col = len(occupied_cells[0])
        verticals, horizontals = stock["grid"]

        # Loop through each cell of the grid and find the cluster of cells that are not occupied,
        # where the item can be placed.
        for i in range(num_row):
            for j in range(num_col):
                if not occupied_cells[i][j]:
                    # Check if there is enough space to the right of the current cell.
                    right_edge = None
                    for k in range(j + 1, num_col + 1):
                        if occupied_cells[i][k - 1]:
                            break

                        if verticals[k] >= verticals[j] + product_width:
                            right_edge = k
                            break

                    # Check if there is enough space above the current cell.
                    if right_edge != None:
                        for k in range(i + 1, num_row + 1):
                            if occupied_cells[k - 1][j]:
                                break

                            if horizontals[k] >= horizontals[i] + product_height:
                                obstacle = False
                                for check_space_i in range(i, k):
                                    for check_space_j in range(j, right_edge):
                                        if occupied_cells[check_space_i][check_space_j]:
                                            obstacle = True
                                            break
                                    if obstacle:
                                        break
                                
                                if obstacle:
                                    break
                                
                                # Update the grid after the item is placed.
                                if verticals[j] + product_width < verticals[right_edge]:
                                    verticals.add(verticals[j] + product_width)
                                    for row in occupied_cells:
                                        row.insert(right_edge, row[right_edge - 1])

                                if horizontals[i] + product_height < horizontals[k]:
                                    horizontals.add(horizontals[i] + product_height)
                                    occupied_cells.insert(k, copy.deepcopy(occupied_cells[k - 1]))

                                # Update the occupied cells after the item is placed.
                                for m in range(i, k):
                                    for n in range(j, right_edge):
                                        occupied_cells[m][n] = True
                                
                                # Place the item.
                                stock["products"].append(
                                    [verticals[j], horizontals[i], product_width, product_height]
                                )

                                # Update inner bound.
                                if stock["right_bound"] < verticals[j] + product_width:
                                    stock["right_bound"] = verticals[j] + product_width
                                if stock["top_bound"] < horizontals[i] + product_height:
                                    stock["top_bound"] = horizontals[i] + product_height
                                return verticals[j], horizontals[i]

    @staticmethod
    def tighten(stocks, stocks_sort):
        # Sort the stocks array in descending order of wasted areas.
        wasted_indices = np.arange(len(stocks))
        wasted_indices, _ = zip(
            *sorted(
                zip(wasted_indices, stocks), 
                key = lambda x: float("inf") if x[1]["right_bound"] * x[1]["top_bound"] == 0 
                else x[1]["right_bound"] * x[1]["top_bound"] - x[1]["width"] * x[1]["height"]
            )
        )

        # Iterate through the most wasted stocks and move all their products to other stocks if smaller.
        for stock_index in wasted_indices:
            stock = stocks[stock_index]
            # Start by checking the smallest stocks first.
            for i in stocks_sort[::-1]:
                replace_stock = stocks[i]
                if replace_stock["top_bound"] * replace_stock["right_bound"] == 0 and \
                   replace_stock["width"] >= stock["right_bound"] and replace_stock["height"] >= stock["top_bound"] and \
                   replace_stock["width"] * replace_stock["height"] < stock["width"] * stock["height"]:
                    replace_stock["products"] = stock["products"]
                    replace_stock["right_bound"] = stock["right_bound"]
                    replace_stock["top_bound"] = stock["top_bound"]

                    stock["products"] = []
                    stock["right_bound"] = 0
                    stock["top_bound"] = 0
                    break
