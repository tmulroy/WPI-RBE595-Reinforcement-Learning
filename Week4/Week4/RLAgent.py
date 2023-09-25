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

        self.__policy = self.set_initial_policy()

    def run(self):
        pass

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

    def set_initial_policy(self):
        '''
        Sets initial agent policy to all 8 actions
        :return: policy: dict{(x,y)} = [all 8 action]
        '''
        policy = {}
        it = np.nditer(self.world.grid, flags=['multi_index'])
        for state in it:
            key = (it.multi_index[0], it.multi_index[1])
            value = self.actions
            policy[key] = value
        return policy

    def value_iteration(self):
        '''
        Performs Value Iteration according to Sutton, p.83, Eq. 4.10
        :return:
        '''
        it = np.nditer(self.state_values, flags=['multi_index'])
        v = self.state_values
        for s in it:
            # Get policy for each state to get s_primes
            s_primes = []




if __name__ == '__main__':
    world = World()
    world.show()
    agent = RLAgent(world)
    agent.value_iteration()
