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