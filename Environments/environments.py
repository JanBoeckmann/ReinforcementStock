from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np

number_of_periods_until_perish = 3
number_of_periods = 100
demand_mean = 11

gain_per_sales = 10
loss_per_perished_product = 30

inital_inventory = 26

case_size = 12

maximum_number_of_cases_to_order = 4


def update_state(action, state, number_of_periods):
    # demand and inventory update logic
    demand = np.random.poisson(demand_mean)

    original_demand = demand

    # update inventory with non-picking
    for s in range(number_of_periods_until_perish):
        if s < number_of_periods_until_perish - 1:
            reverse_index = -1 - s
        else:
            reverse_index = 0
        if state[reverse_index] > demand:
            state[reverse_index] -= demand
            demand = 0
        else:
            demand -= state[reverse_index]
            state[reverse_index] = 0

    number_of_periods -= 1


    reward = gain_per_sales * (original_demand - demand) - loss_per_perished_product * state[number_of_periods_until_perish - 1]
    state = np.roll(state, 1)  # Shift all elements to the right
    state[0] = action * case_size  # Set the new first element 

    # print("demand:", original_demand)
    done = number_of_periods <= 0  
    truncated = False
    info = {}

    return reward, done, truncated, info, state, number_of_periods

class EnvironmentStock(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(maximum_number_of_cases_to_order) # number of cases to order
        self.observation_space = Box(low=np.array([0 for _ in range(number_of_periods_until_perish)]), high=np.array([case_size * maximum_number_of_cases_to_order for i in range(number_of_periods_until_perish)]), dtype=np.float32)
        self.state = [0 for _ in range(number_of_periods_until_perish)]
        self.state[0] = 26
        self.number_of_periods = number_of_periods  

    def step(self, action):
        reward, done, truncated, info, self.state, self.number_of_periods = update_state(action, self.state, self.number_of_periods)
        return np.array(self.state, dtype=np.float32), reward, done, truncated, info  # Ensure correct format

    def reset(self, seed=None, options=None):
        self.number_of_periods = 100 
        self.state = [0 for _ in range(number_of_periods_until_perish)]
        self.state[0] = inital_inventory 
        self.state = np.array(self.state, dtype=np.float32)
        observation = np.array(self.state, dtype=np.float32)
        return observation, {}

    
