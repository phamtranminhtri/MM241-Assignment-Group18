from policy import Policy
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self):
        pass  # No longer need to initialize product_order

    def get_action(self, observation, info):
        # Get all products with quantity > 0
        products_with_indices = [
            (idx, prod) for idx, prod in enumerate(observation["products"]) if prod["quantity"] > 0
        ]

        # If no products are left to place, return no action
        if not products_with_indices:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Sort products by decreasing height (size[1])
        sorted_products = sorted(
            products_with_indices, key=lambda x: x[1]["size"][1], reverse=True
        )

        # Iterate over sorted products
        for prod_idx, prod in sorted_products:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            # Try to place the product into the first stock where it fits
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                if stock_w < prod_w or stock_h < prod_h:
                    continue
                # Find position where the product can fit
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            # Return action (environment updates quantities)
                            return {"stock_idx": i, "size": prod_size, "position": (x, y)}
        # If no products can be placed, return no action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    

class BFDPolicy(Policy):
    def __init__(self):
        self.stock_areas = None
        self.occupied_areas = None
        self.skylines = None  # Track height profiles
        self.sorted_products = None
        
    def _initialize_skylines(self, stocks):
        """Initialize skyline height profiles for each stock"""
        self.skylines = []
        self.stock_areas = []
        self.occupied_areas = []
        
        for stock in stocks:
            w, h = self._get_stock_size_(stock)
            self.skylines.append(np.zeros(w, dtype=np.int32))  # Height profile
            self.stock_areas.append(w * h)
            self.occupied_areas.append(0)

    def _find_position(self, stock, skyline, width, height):
        """Find first position using skyline algorithm"""
        stock_w = len(skyline)
        min_height = float('inf')
        best_x = -1
        
        # Use sliding window to find minimum height position
        for x in range(stock_w - width + 1):
            local_height = np.max(skyline[x:x + width])
            if local_height + height <= self._get_stock_size_(stock)[1]:
                # Check if position is available
                if self._can_place_(stock, (x, local_height), (width, height)):
                    if local_height < min_height:
                        min_height = local_height
                        best_x = x
                        # Early stopping if we found ground level
                        if local_height == 0:
                            break
                            
        return best_x, min_height if best_x != -1 else (-1, -1)

    def get_action(self, observation, info):
        # Reset state on new episode
        if all(np.all(stock <= -1) for stock in observation["stocks"]):
            self._initialize_skylines(observation["stocks"])
            # Sort products by area and height
            products_with_indices = [
                (idx, prod) for idx, prod in enumerate(observation["products"])
                if prod["quantity"] > 0
            ]
            self.sorted_products = sorted(
                products_with_indices,
                key=lambda x: (-x[1]["size"][0] * x[1]["size"][1], -x[1]["size"][1])
            )
        
        # Get next product to place
        valid_products = [
            (idx, prod) for idx, prod in self.sorted_products
            if observation["products"][idx]["quantity"] > 0
        ]
        
        if not valid_products:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
        prod_idx, prod = valid_products[0]
        prod_w, prod_h = prod["size"]
        prod_area = prod_w * prod_h
        
        # Find best fitting stock using skyline
        best_stock_idx = -1
        best_pos = None
        min_waste = float('inf')
        
        # Only check stocks with enough area
        viable_stocks = [
            (i, stock) for i, stock in enumerate(observation["stocks"])
            if self.stock_areas[i] - self.occupied_areas[i] >= prod_area
        ]
        
        for i, stock in viable_stocks:
            x, y = self._find_position(stock, self.skylines[i], prod_w, prod_h)
            if x != -1:
                waste = self.stock_areas[i] - self.occupied_areas[i] - prod_area
                if waste < min_waste:
                    min_waste = waste
                    best_stock_idx = i
                    best_pos = (x, y)
                    # Early stopping if perfect fit
                    if waste == 0:
                        break
                        
        if best_stock_idx != -1:
            # Update skyline and occupied area
            self.skylines[best_stock_idx][best_pos[0]:best_pos[0] + prod_w] = best_pos[1] + prod_h
            self.occupied_areas[best_stock_idx] += prod_area
            return {
                "stock_idx": best_stock_idx,
                "size": prod["size"],
                "position": best_pos
            }
            
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    
from queue import PriorityQueue
from dataclasses import dataclass
import copy

@dataclass
class BBNode:
    remaining_quantities: dict  # {prod_idx: remaining_qty}
    used_stocks: set            # Set of stock indices used
    value: float                # Total area of placed products
    actions: list               # List of actions taken to reach this node

class BranchAndBoundPolicy(Policy):
    def __init__(self):
        self.best_value = float('-inf')
        self.pq = PriorityQueue()
        self.initialized = False
        self.action_plan = []
    
    def _calculate_lower_bound(self, node, observation):
        """Calculate lower bound based on placed area."""
        total_area = sum(
            prod["size"][0] * prod["size"][1] * prod["quantity"]
            for prod in observation["products"]
        )
        remaining_area = sum(
            observation["products"][idx]["size"][0] * observation["products"][idx]["size"][1] * qty
            for idx, qty in node.remaining_quantities.items()
        )
        placed_area = total_area - remaining_area
        return placed_area  # Maximize placed area
    
    def get_action(self, observation, info):
        # Initialize on new episode
        if not self.initialized or all(np.all(stock <= -1) for stock in observation["stocks"]):
            self.initialized = True
            self.best_value = float('-inf')
            self.pq = PriorityQueue()
            self.action_plan = []

            initial_quantities = {
                idx: prod["quantity"]
                for idx, prod in enumerate(observation["products"])
                if prod["quantity"] > 0
            }

            initial_node = BBNode(
                remaining_quantities=initial_quantities,
                used_stocks=set(),
                value=0,
                actions=[]
            )

            lb = self._calculate_lower_bound(initial_node, observation)
            self.pq.put((-lb, initial_node))

            # Perform the Branch and Bound search to find the best action plan
            while not self.pq.empty():
                _, node = self.pq.get()

                # Check if all products are placed
                if all(qty == 0 for qty in node.remaining_quantities.values()):
                    if node.value > self.best_value:
                        self.best_value = node.value
                        self.action_plan = node.actions
                    continue

                # Select the next product to place (e.g., the one with the highest area)
                prod_idx, qty = max(node.remaining_quantities.items(), key=lambda x: observation["products"][x[0]]["size"][0] * observation["products"][x[0]]["size"][1])
                prod = observation["products"][prod_idx]
                prod_w, prod_h = prod["size"]

                # Try to place in each stock
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Try all positions in the stock
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                # Create new node
                                new_remaining_quantities = node.remaining_quantities.copy()
                                new_remaining_quantities[prod_idx] -= 1

                                new_used_stocks = node.used_stocks.copy()
                                new_used_stocks.add(stock_idx)

                                new_value = node.value + prod_w * prod_h

                                new_actions = node.actions.copy()
                                new_actions.append({
                                    "stock_idx": stock_idx,
                                    "size": prod["size"],
                                    "position": (x, y)
                                })

                                child_node = BBNode(
                                    remaining_quantities=new_remaining_quantities,
                                    used_stocks=new_used_stocks,
                                    value=new_value,
                                    actions=new_actions
                                )

                                lb = self._calculate_lower_bound(child_node, observation)
                                self.pq.put((-lb, child_node))

            # After search, the best action plan is stored in self.action_plan

        # Return the next action from the action plan
        if self.action_plan:
            return self.action_plan.pop(0)
        else:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        

class Policy2312291(Policy):
    def __init__(self):
        self.free_rectangles = {}
        self.previous_stock_states = None

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        
        # Reset state for new episode
        if self._is_new_episode(stocks):
            self.free_rectangles = {}
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                self.free_rectangles[stock_idx] = [{
                    "x": 0, "y": 0, "w": stock_w, "h": stock_h
                }]
            self.previous_stock_states = [stock.copy() for stock in stocks]

        products = observation["products"]
        available_products = [
            prod for prod in products if prod["quantity"] > 0
        ]

        if not available_products:
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

        # Sort by descending area
        available_products.sort(
            key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )

        best_placement = None
        min_waste = float('inf')

        # Try each product
        for prod in available_products:
            prod_w, prod_h = prod["size"]

            # Try each stock
            for stock_idx, stock in enumerate(stocks):
                free_rects = self.free_rectangles[stock_idx]
                
                # Try each free rectangle
                for rect in free_rects:
                    if prod_w <= rect["w"] and prod_h <= rect["h"]:
                        waste = rect["w"] * rect["h"] - prod_w * prod_h
                        if waste < min_waste:
                            min_waste = waste
                            best_placement = {
                                "stock_idx": stock_idx,
                                "position": (rect["x"], rect["y"]),
                                "size": prod["size"]
                            }
                            if waste == 0:  # Perfect fit
                                break
                if min_waste == 0:
                    break
            if min_waste == 0:
                break

        if best_placement is not None:
            stock_idx = best_placement["stock_idx"]
            pos_x, pos_y = best_placement["position"]
            prod_w, prod_h = best_placement["size"]
            
            # Update free rectangles
            self._split_free_rectangle(stock_idx, pos_x, pos_y, prod_w, prod_h)
            
            return {
                "stock_idx": stock_idx,
                "size": best_placement["size"],
                "position": best_placement["position"]
            }

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _split_free_rectangle(self, stock_idx, x, y, w, h):
        new_free_rectangles = []
        for rect in self.free_rectangles[stock_idx]:
            if not self._rects_overlap(x, y, w, h, rect["x"], rect["y"], rect["w"], rect["h"]):
                new_free_rectangles.append(rect)
            else:
                self._split_rectangles(rect, x, y, w, h, new_free_rectangles)
        self.free_rectangles[stock_idx] = self._prune_free_rectangles(new_free_rectangles)

    def _split_rectangles(self, rect, x, y, w, h, new_free_rectangles):
        # Left split
        if rect["x"] < x:
            new_rect = {
                "x": rect["x"],
                "y": rect["y"],
                "w": x - rect["x"],
                "h": rect["h"]
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

        # Right split
        if x + w < rect["x"] + rect["w"]:
            new_rect = {
                "x": x + w,
                "y": rect["y"],
                "w": (rect["x"] + rect["w"]) - (x + w),
                "h": rect["h"]
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

        # Top split
        if rect["y"] < y:
            new_rect = {
                "x": rect["x"],
                "y": rect["y"],
                "w": rect["w"],
                "h": y - rect["y"]
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

        # Bottom split
        if y + h < rect["y"] + rect["h"]:
            new_rect = {
                "x": rect["x"],
                "y": y + h,
                "w": rect["w"],
                "h": (rect["y"] + rect["h"]) - (y + h)
            }
            if new_rect["w"] > 0 and new_rect["h"] > 0:
                new_free_rectangles.append(new_rect)

    def _prune_free_rectangles(self, free_rectangles):
        pruned = []
        for rect in free_rectangles:
            if not any(self._is_rect_inside(rect, other) 
                      for other in free_rectangles if rect != other):
                pruned.append(rect)
        return pruned

    def _rects_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or 
                   y1 + h1 <= y2 or y2 + h2 <= y1)

    def _is_rect_inside(self, inner, outer):
        return (inner["x"] >= outer["x"] and
                inner["y"] >= outer["y"] and
                inner["x"] + inner["w"] <= outer["x"] + outer["w"] and
                inner["y"] + inner["h"] <= outer["y"] + outer["h"])

    def _is_new_episode(self, stocks):
        if self.previous_stock_states is None:
            return True
        return all(np.all(stock <= -1) for stock in stocks)