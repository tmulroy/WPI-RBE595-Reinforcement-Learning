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
        self.env_probabilities = {
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

    def initialize(self):
        '''
        Sets initial agent policy to all 8 actions
        :return: policy,s_vals: dict{(x,y)} = [all 8 action], dict{(x,y)} = 0
        '''
        policy = {}
        s_vals = {}
        it = np.nditer(self.world.grid, flags=['multi_index'])
        for state in it:
            key = (it.multi_index[0], it.multi_index[1])
            policy_value = self.actions
            policy[key] = policy_value
            s_vals[key] = 0
        # print(f's_vals: {s_vals}')

        # Need to account for world borders in policy
        rows,cols = self.world.grid.shape
        for key,value in policy.items():
            if key[0] == 0 and 0 < key[1] < cols-1: # Top Row
                policy[key] = ['east','south','west','south-east','south-west']
            elif key[0] == rows-1: # Bottom Row
                policy[key] = ['north','east','west','north-east','north-west']
            elif key[1] == 0 and 0 < key[0] < rows-1: # Leftmost Column
                policy[key] = ['north','east','south','north-east','south-east']
            elif key[1] == cols-1: # Rightmost Column
                policy[key] = ['north','south','west','north-west','south-west']
            else:
                pass

        # Handle Corners
        # Top Left
        policy[(0,0)] = ['east','south-east','south']
        # Top Right
        policy[(0,cols-1)] = ['south','south-west','west']
        # Bottom Left
        policy[(rows-1, 0)] = ['north','north-east','east']
        # Bottom Right
        policy[(rows-1, cols-1)] = ['north','west','north-west']

        return policy,s_vals

    def value_iteration(self):
        '''
        Performs Value Iteration according to Sutton, p.83, Eq. 4.10
        :return:
        '''
        it = np.nditer(self.state_values, flags=['multi_index'])
        v = self.state_values

        # Iterate through every state in state_values
        for s in it:
            # Get policy for each state to get s_primes
            s_primes_val = []
            s_primes_reward = []

            # Get actions associated with policy for that state
            s_actions = self.policy[(it.multi_index[0], it.multi_index[1])]

            # Map string actions to grid coordinates (dict keys)
            north = (it.multi_index[0]-1, it.multi_index[1])
            east = (it.multi_index[0], it.multi_index[1]+1)
            south = (it.multi_index[0]+1, it.multi_index[1])
            west = (it.multi_index[0], it.multi_index[1]-1)
            north_east = (it.multi_index[0]-1, it.multi_index[1]+1)
            south_east = (it.multi_index[0]+1, it.multi_index[1]+1)
            south_west = (it.multi_index[0]+1, it.multi_index[1]-1)
            north_west = (it.multi_index[0]-1, it.multi_index[1]-1)

            # Get s' value and reward
            print(f'state: ({it.multi_index[0]},{it.multi_index[1]})')
            for action in s_actions:

                if action == 'north':
                    s_primes_val.append(self.state_values_dict[north])
                    pass
                elif action == 'east':
                    s_primes_val.append(self.state_values_dict[east])
                    pass
                elif action == 'south':
                    s_primes_val.append(self.state_values_dict[south])
                    pass
                elif action == 'west':
                    s_primes_val.append(self.state_values_dict[west])
                    pass
                elif action == 'north-east':
                    s_primes_val.append(self.state_values_dict[north_east])
                    pass
                elif action == 'south-east':
                    s_primes_val.append(self.state_values_dict[south_east])
                    pass
                elif action == 'south-west':
                    s_primes_val.append(self.state_values_dict[south_west])
                    pass
                elif action == 'north-west':
                    s_primes_val.append(self.state_values_dict[north_west])
                    pass
                else:
                    pass


            # print(f's_primes: {s_primes_val}')
            # next_state_val = self.reward + self.gamma*




if __name__ == '__main__':
    world = World()
    world.show()
    agent = RLAgent(world)
    agent.value_iteration()
    print(agent.policy)
