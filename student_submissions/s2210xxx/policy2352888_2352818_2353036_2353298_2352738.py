from policy import Policy
from typing import Dict, List

class Policy2352888_2352818_2353036_2353298_2352738(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.policy_id = policy_id
        if policy_id == 1:
            self.policy = ColumnGenerationPolicy()
        elif policy_id == 2:
            self.policy = BestFitPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
    
class ColumnGenerationPolicy(Policy):
    def __init__(self, 
                 min_waste_threshold: float = 0.05, 
                 max_iterations: int = 100):
        """
        Column Generation Policy for Cutting Stock Problem
        
        Optimizes for multi-order placement and minimized waste
        
        Args:
            min_waste_threshold (float): Minimum acceptable waste ratio
            max_iterations (int): Maximum pattern generation iterations
        """
        self.min_waste_threshold = min_waste_threshold
        self.max_iterations = max_iterations
        self.cutting_patterns = []

    def _calculate_waste(self, 
                          stock_size: List[int], 
                          products: List[Dict]) -> float:
        """
        Calculate waste for a potential cutting pattern
        
        Args:
            stock_size (List[int]): Dimensions of stock material
            products (List[Dict]): List of products to place
        
        Returns:
            float: Waste ratio (0-1)
        """
        total_product_area = sum(
            prod['size'][0] * prod['size'][1] * prod.get('quantity', 1) 
            for prod in products
        )
        
        # Stock area
        stock_area = stock_size[0] * stock_size[1]
        
        # Waste calculation
        waste_ratio = 1 - (total_product_area / stock_area)
        return max(0, waste_ratio)

    def _generate_comprehensive_cutting_pattern(self, 
                                                stock_size: List[int], 
                                                available_products: List[Dict]) -> Dict:
        """
        Generate a comprehensive cutting pattern with multiple products
        
        Args:
            stock_size (List[int]): Stock material dimensions
            available_products (List[Dict]): Available products
        
        Returns:
            Dict: Cutting pattern with multiple products and minimal waste
        """
        best_pattern = None
        min_waste = float('inf')
        
        # Sort products by area (largest first) for better packing
        sorted_products = sorted(
            [p for p in available_products if p['quantity'] > 0], 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        # Try multiple placement strategies
        for placement_strategy in [False, True]:  # With and without rotation
            for num_products in range(1, len(sorted_products) + 1):
                for product_subset in self._generate_product_combinations(
                    sorted_products, num_products
                ):
                    pattern = self._try_pack_multiple_products(
                        stock_size, 
                        product_subset, 
                        rotate_allowed=placement_strategy
                    )
                    
                    if pattern:
                        waste = self._calculate_waste(stock_size, pattern['products'])
                        
                        if waste < min_waste:
                            min_waste = waste
                            best_pattern = pattern
                        
                        if waste <= self.min_waste_threshold:
                            return best_pattern
        
        return best_pattern

    def _try_pack_multiple_products(self, 
                                    stock_size: List[int], 
                                    products: List[Dict], 
                                    rotate_allowed: bool = True) -> Dict:
        """
        Advanced multi-product packing algorithm
        
        Args:
            stock_size (List[int]): Stock material dimensions
            products (List[Dict]): Products to pack
            rotate_allowed (bool): Allow product rotation
        
        Returns:
            Dict: Packing configuration or None
        """
        placed_products = []
        remaining_space = list(stock_size)
        
        sorted_products = sorted(
            products, 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        for product in sorted_products:
            orientations = [
                product['size'],
                product['size'][::-1] if rotate_allowed else None
            ]
            orientations = [o for o in orientations if o is not None]
            
            placed = False
            for orientation in orientations:
                if (orientation[0] <= remaining_space[0] and 
                    orientation[1] <= remaining_space[1]):
                    placed_products.append({
                        **product,
                        'size': orientation,
                        'rotated': orientation != product['size']
                    })
                    
                    remaining_space[0] -= orientation[0]
                    
                    if remaining_space[0] == 0:
                        remaining_space[0] = stock_size[0]
                        remaining_space[1] -= orientation[1]
                    
                    placed = True
                    break
            
            if not placed:
                break
        
        return {
            'products': placed_products,
            'total_waste': self._calculate_waste(stock_size, placed_products)
        } if placed_products else None
    def _compare_product_sizes(self, prod1, prod2):
        """
        Compare product sizes accounting for NumPy array comparison
        
        Args:
            prod1 (Dict): First product
            prod2 (Dict): Second product
        
        Returns:
            bool: True if product sizes match, False otherwise
        """
        size1 = tuple(prod1['size'])
        size2 = tuple(prod2['size'])
        
        return (size1 == size2) or (size1 == size2[::-1])
    def get_action(self, observation: Dict, info: Dict) -> Dict:
        """
        Select product placement using column generation strategy,
        attempting to fill each stock area completely with multiple products
        
        Args:
            observation (Dict): Environment observation
            info (Dict): Additional environment information
        
        Returns:
            Dict: Placement action or None if no placement is possible
        """
        if not hasattr(self, '_current_stock_idx'):
            self._current_stock_idx = 0
            self._occupied_positions = {}
        
        list_prods = observation["products"]
        stocks = observation["stocks"]
        
        valid_products = [prod for prod in list_prods if prod["quantity"] > 0]
        if not valid_products:
            return None
        
        for stock_idx in range(self._current_stock_idx, len(stocks)):
            stock = stocks[stock_idx]
            
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_size = [stock_w, stock_h]
            
            if stock_idx not in self._occupied_positions:
                self._occupied_positions[stock_idx] = set()
            
            def is_position_available(pos, size):
                for x in range(pos[0], pos[0] + size[0]):
                    for y in range(pos[1], pos[1] + size[1]):
                        if (x, y) in self._occupied_positions[stock_idx]:
                            return False
                return True
            
            cutting_pattern = self._generate_cutting_pattern(
                stock_size, 
                valid_products
            )
            
            if cutting_pattern and cutting_pattern.get('products'):
                for first_product in cutting_pattern['products']:
                    prod_size = first_product['size']
                    
                    if first_product.get('rotated', False):
                        prod_size = prod_size[::-1]
                    
                    placement_pos = None
                    for y in range(stock_h - prod_size[1] + 1):
                        for x in range(stock_w - prod_size[0] + 1):
                            if is_position_available((x, y), prod_size):
                                placement_pos = (x, y)
                                break
                        if placement_pos:
                            break
                    
                    if placement_pos is None:
                        continue
                    
                    placement = {
                        'stock_idx': stock_idx,
                        'size': prod_size,
                        'position': placement_pos
                    }
                    
                    if self._can_place_(stock, placement_pos, prod_size):
                        for x in range(placement_pos[0], placement_pos[0] + prod_size[0]):
                            for y in range(placement_pos[1], placement_pos[1] + prod_size[1]):
                                self._occupied_positions[stock_idx].add((x, y))
                        
                        valid_products = [
                            p for p in valid_products 
                            if not self._compare_product_sizes(p, first_product)
                        ]
                        
                        self._current_stock_idx = stock_idx
                        
                        return placement
            
            if not valid_products:
                break
        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def _generate_product_combinations(self, 
                                       products: List[Dict], 
                                       num_products: int) -> List[List[Dict]]:
        """
        Generate combinations of products with unique sizes
        
        Args:
            products (List[Dict]): Available products
            num_products (int): Number of products to combine
        
        Returns:
            List[List[Dict]]: Unique product combinations
        """
        if num_products > len(products):
            return []
        
        combinations = []
        
        def backtrack(start: int, current_combo: List[Dict]):
            if len(current_combo) == num_products:
                combinations.append(current_combo.copy())
                return
            
            for i in range(start, len(products)):
                if not any(
                    current_combo and 
                    tuple(p['size']) == tuple(products[i]['size']) 
                    for p in current_combo
                ):
                    current_combo.append(products[i])
                    
                    backtrack(i + 1, current_combo)
                    
                    current_combo.pop()
        
        backtrack(0, [])
        
        return combinations
    def _generate_cutting_pattern(self, 
                               stock_size: List[int], 
                               available_products: List[Dict]) -> Dict:
        """
        Generate an optimized cutting pattern with minimal waste
        
        Args:
            stock_size (List[int]): Stock material dimensions
            available_products (List[Dict]): Available products to place
        
        Returns:
            Dict: Optimized cutting pattern or None if no pattern found
        """
        sorted_products = sorted(
            available_products, 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        packing_strategies = [
            (False, False),  # No rotation
            (True, False),   # Rotate products
            (False, True),   # Prioritize height
            (True, True)     # Rotate and prioritize height
        ]
        
        best_pattern = None
        min_waste = float('inf')
        
        for rotate_products, prioritize_height in packing_strategies:
            current_stock_size = list(stock_size)
            
            placed_products = []
            remaining_space = list(current_stock_size)
            
            current_products = sorted_products.copy()
            
            while current_products:
                packed = False
                if prioritize_height:
                    current_products.sort(key=lambda p: p['size'][1], reverse=True)
                
                for idx, product in enumerate(current_products):
                    prod_size = product['size']
                    if rotate_products:
                        prod_size = prod_size[::-1]
                    
                    if (prod_size[0] <= remaining_space[0] and 
                        prod_size[1] <= remaining_space[1]):
                        
                        placed_products.append({
                            **product,
                            'rotated': rotate_products
                        })
                        
                        remaining_space[0] -= prod_size[0]
                        
                        if remaining_space[0] == 0:
                            remaining_space[0] = current_stock_size[0]
                            remaining_space[1] -= prod_size[1]
                        
                        current_products.pop(idx)
                        
                        packed = True
                        break
                
                if not packed:
                    break
            
            if placed_products:
                waste = self._calculate_waste(current_stock_size, placed_products)
                
                if waste < min_waste:
                    min_waste = waste
                    best_pattern = {
                        'products': placed_products,
                        'rotated': rotate_products,
                        'waste': waste
                    }
                
                if waste <= self.min_waste_threshold:
                    return best_pattern
        
        return best_pattern
class BestFitPolicy(Policy):
    
    def __init__(self):
        pass

    def get_action(self, observation, info):
        # Lấy danh sách sản phẩm và sắp xếp theo diện tích giảm dần
        sorted_products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        # Duyệt qua từng sản phẩm đã sắp xếp
        for product in sorted_products:
            if product["quantity"] > 0:  # Chỉ xử lý nếu còn sản phẩm
                prod_size = product["size"]
                prod_w, prod_h = prod_size

                best_fit = None
                best_fit_position = None
                min_waste = float('inf')  # Giá trị thừa không gian nhỏ nhất

                # Duyệt qua từng tấm gỗ (stock)
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Kiểm tra không xoay sản phẩm
                    if stock_w >= prod_w and stock_h >= prod_h:
                        position = self._find_first_fit_(stock, (prod_w, prod_h))
                        if position:
                            waste = (stock_w - prod_w) * (stock_h - prod_h)
                            if waste < min_waste:
                                best_fit = (stock_idx, prod_size, position)
                                min_waste = waste

                    # Kiểm tra xoay 90 độ
                    if stock_w >= prod_h and stock_h >= prod_w:
                        position = self._find_first_fit_(stock, (prod_h, prod_w))
                        if position:
                            waste = (stock_w - prod_h) * (stock_h - prod_w)
                            if waste < min_waste:
                                best_fit = (stock_idx, prod_size[::-1], position)
                                min_waste = waste

                # Nếu tìm được tấm gỗ phù hợp nhất, trả về hành động
                if best_fit:
                    stock_idx, prod_size, position = best_fit
                    return {"stock_idx": stock_idx, "size": prod_size, "position": position}

        # Nếu không tìm được vị trí phù hợp
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_first_fit_(self, stock, prod_size):
        """Tìm vị trí đầu tiên có thể đặt sản phẩm trong tấm gỗ."""
        prod_w, prod_h = prod_size
        stock_w, stock_h = stock.shape

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None
