import numpy as np

class Policy:
    def __init__(self):
        self.reorder_treshold = 26

    def __str__(self):
        return f"Policy Name: {self.name}, Description: {self.description}"

    def __repr__(self):
        return f"Policy({self.name}, {self.description})"
    
    def get_action(self, state):
        stock = np.sum(state)
        if stock < self.reorder_treshold:
            return [1], None
        else:
            return [0], None