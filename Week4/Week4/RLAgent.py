import numpy as np
from world import World

class RLAgent:
    def __init__(self, world, gamma=0.95, start_state=(7,25)):
        self.__gamma = gamma
        self.__deterministic = True
        self.__actions = ['north','east','south','west',
                          'north-east','south-east','south-west','north-west']
        self.__start_state = start_state
        self.__delta = 0
        self.world = world
        self.state_values = np.zeros((self.world.grid.shape[0], self.world.grid.shape[1]))
        self.__env_probabilities = {
            'north': 1,
            'east': 1,
            'south': 1,
            'west': 1,
            'north-west': 1,
            'north-east': 1,
            'south-east': 1,
            'south-west': 1}

        self.__policy, self.__state_values_dict = self.initialize()

    def run(self):
        pass

    @property
    def gamma(self):
        return self.__gamma

    @property
    def state_values_dict(self):
        return self.__state_values_dict

    @property
    def deterministic(self):
        return self.__deterministic

    @property
    def actions(self):
        return self.__actions

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, new_policy):
        self.__policy = new_policy
        pass

    @property
    def env_probabilities(self):
        return self.__env_probabilities

    def initialize(self):
        '''
        Sets initial agent policy
        Sets initial state values to 0
        :return: policy,s_vals: dict{(x,y)} = [all 8 action], dict{(x,y)} = 0
        '''
        policy = {}
        s_vals = {}
        it = np.nditer(self.world.grid, flags=['multi_index'])
        rows,cols = self.world.grid.shape

        # Set each State Value to 0
        # Set policy to map every state to all actions
        for state in it:
            key = (it.multi_index[0], it.multi_index[1])
            policy_value = self.actions
            policy[key] = policy_value
            s_vals[key] = 0

        # Account for World Borders in Policy
        for key,value in policy.items():
            if key[0] == 0 and 0 < key[1] < cols-1: # Top Row
                policy[key] = ['east','south','west','south-east','south-west']
            elif key[0] == rows-1 and 0 < key[1] < cols-1: # Bottom Row
                policy[key] = ['north','east','west','north-east','north-west']
            elif key[1] == 0 and 0 < key[0] < rows-1: # Leftmost Column
                policy[key] = ['north','east','south','north-east','south-east']
            elif key[1] == cols-1 and 0 < key[0] < rows-1: # Rightmost Column
                policy[key] = ['north','south','west','north-west','south-west']
            else:
                pass

        # Account for World Corners in Policy
        policy[(0,0)] = ['east','south-east','south']  # Top Left
        policy[(0,cols-1)] = ['south','south-west','west']  # Top Right
        policy[(rows-1, 0)] = ['north','north-east','east']  # Bottom Left
        policy[(rows-1, cols-1)] = ['north','west','north-west']  # Bottom Right

        return policy,s_vals

    def value_iteration(self):
        '''
        Performs Value Iteration according to Sutton, p.83, Eq. 4.10
        :return:
        '''
        it = np.nditer(self.state_values, flags=['multi_index'])
        next_state_val = 0
        future_rewards = {}

        # Iterate through every state in state_values
        for state in it:
            s_primes_state_vals = {}
            state_idx = (it.multi_index[0], it.multi_index[1])
            print(f'state: {state_idx}')
            # Variables we need to solve Bellman Optimal Value Equation:
            curr_state_val = next_state_val

            # Get actions associated with policy for that state
            state_actions = self.policy[state_idx]

            # Map string actions to grid coordinates (dict keys)
            coords = {
                'north': (it.multi_index[0]-1, it.multi_index[1]),
                'east': (it.multi_index[0], it.multi_index[1]+1),
                'south': (it.multi_index[0]+1, it.multi_index[1]),
                'west': (it.multi_index[0], it.multi_index[1]-1),
                'north-east': (it.multi_index[0]-1, it.multi_index[1]+1),
                'south-east': (it.multi_index[0]+1, it.multi_index[1]+1),
                'south-west': (it.multi_index[0]+1, it.multi_index[1]-1),
                'north-west': (it.multi_index[0]-1, it.multi_index[1]-1)

            }

            # Get s' value and reward
            sum = 0
            for action in state_actions:
                # Sum up r+gamma*V(s')
                # sum += 1*(immediate_reward[state_idx] + self.gamma*self.state_values_dict[action])
                immediate_reward = self.world.rewards[coords[action]]
                print(f'action: {action}, reward: {immediate_reward}')
                s_primes_state_vals[action] = self.env_probabilities[action] * (immediate_reward + self.gamma*self.state_values_dict[coords[action]])

            # Calculate next_state_value
            print(f's_prime: {s_primes_state_vals}\n')
            # next_state_val = self.reward + self.gamma*
            # future_rewards[]



if __name__ == '__main__':
    world = World()
    # world.show()
    print(world.rewards)
    agent = RLAgent(world)
    agent.value_iteration()
