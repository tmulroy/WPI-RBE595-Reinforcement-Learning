import numpy as np
from world import World


class RLAgent:
    def __init__(self, world, theta, deterministic=True, gamma=0.95, start_state=(7,25)):
        self.world = world
        self.__theta = theta
        self.__gamma = gamma
        self.__deterministic = True
        self.__actions = ['north','east','south','west',
                          'north-east','south-east','south-west','north-west']
        self.__start_state = start_state
        self.__env_probabilities = {
            'north': 1,
            'east': 1,
            'south': 1,
            'west': 1,
            'north-west': 1,
            'north-east': 1,
            'south-east': 1,
            'south-west': 1}

        self.__policy, self.__state_values = self.initialize()

    def run(self):
        pass

    @property
    def gamma(self):
        return self.__gamma

    @property
    def theta(self):
        return self.__theta

    @property
    def state_values(self):
        return self.__state_values

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

    @state_values.setter
    def state_values(self, new_state_values):
        self.__state_values = new_state_values

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

    def policy_iteration(self):
        '''
        Performs Policy Iteration according to Sutton, p.80, Eq.
        :return:
        '''
        # Iterate through every state in state_values
        self.policy_evaluation(self.policy, self.state_values)
        # self.policy_improvement()

    def policy_evaluation(self, policy, state_values):
        # for i in range(0, 500):
        delta = 0.0
        current_state = state_values.copy()
        # for i in range(0,1):
        for state_idx, state_value in current_state.items():
            # print(f'i: {i}')
            print(f'state: {state_idx}')
            print(f'  value: {state_value}')
            state_actions = policy[state_idx]
            print(f'  state_actions: {state_actions}')
            neighbor_coords = {
                        'north': (state_idx[0] - 1, state_idx[1]),
                        'east': (state_idx[0], state_idx[1] + 1),
                        'south': (state_idx[0] + 1, state_idx[1]),
                        'west': (state_idx[0], state_idx[1] - 1),
                        'north-east': (state_idx[0] - 1, state_idx[1] + 1),
                        'south-east': (state_idx[0] + 1, state_idx[1] + 1),
                        'south-west': (state_idx[0] + 1, state_idx[1] - 1),
                        'north-west': (state_idx[0] - 1, state_idx[1] - 1)
                    }
            
            probabilistic_sum = 0
            for action in state_actions:
                print(f'    action: {action}')
                immediate_reward = self.world.rewards[neighbor_coords[action]]
                discounted_future_reward = self.gamma*current_state[neighbor_coords[action]]
                print(f'       immediate reward: {immediate_reward}')
                print(f'       discounted future reward: {discounted_future_reward}')
                probabilistic_sum += self.env_probabilities[action]*(immediate_reward + discounted_future_reward)
                print(f'       probabilistic_sum: {probabilistic_sum}')

            state_values[state_idx] = probabilistic_sum
                # diff = abs(current_state[state_idx] - probabilistic_sum)
                # print(f'current_state[state_idx]: {current_state[state_idx]}')
                # print(f'diff: {diff}')
                # delta = max(delta, diff)
                # print(f'delta: {delta}')
                #     current_state[state_idx] = probabilistic_sum
                # if delta < self.theta:
                #     print(f'Policy Iteration Converged on Iteration #{i}')
                #     break


    def value_iteration(self):
        '''
        Performs Value Iteration according to Sutton, p.83, Eq. 4.10
        :return:
        '''
        self.state_values = self.bellman_optimal_state_value_solver()
        self.set_optimal_policy()

    def bellman_optimal_state_value_solver(self):
        temp_state_values = self.state_values
        # Iterate through every state in state_values
        for i in range(0, 500):
            delta = 0.0
            for key in temp_state_values.keys():
                v = temp_state_values[key]
                s_primes_state_vals = {}

                # Get actions associated with policy for that state
                state_actions = self.policy[key]

                # Map string actions to grid coordinates (dict keys)
                coords = {
                    'north': (key[0] - 1, key[1]),
                    'east': (key[0], key[1] + 1),
                    'south': (key[0] + 1, key[1]),
                    'west': (key[0], key[1] - 1),
                    'north-east': (key[0] - 1, key[1] + 1),
                    'south-east': (key[0] + 1, key[1] + 1),
                    'south-west': (key[0] + 1, key[1] - 1),
                    'north-west': (key[0] - 1, key[1] - 1)

                }

                # Calculate Bellman Optimal State-Value Function
                for action in state_actions:
                    immediate_reward = self.world.rewards[coords[action]]
                    s_primes_state_vals[action] = self.env_probabilities[action] * (
                                immediate_reward + self.gamma * temp_state_values[coords[action]])

                # Get Max of Bellman
                temp_state_values[key] = max(s_primes_state_vals.values())
                diff = abs(v - temp_state_values[key])
                delta = max(delta, diff)
            if delta < self.theta:
                print(f'converged on iteration: {i}')
                break
        return temp_state_values

    def set_optimal_policy(self):
        # REFACTOR: abstract to a MDP class
        # REFACTOR: account for multiple actions that are max
        for key in self.state_values.keys():
            state_actions = self.policy[key]
            # Get actions associated with policy for that state
            s_primes_state_vals = {}
            # Map string actions to grid coordinates (dict keys)
            coords = {
                'north': (key[0] - 1, key[1]),
                'east': (key[0], key[1] + 1),
                'south': (key[0] + 1, key[1]),
                'west': (key[0], key[1] - 1),
                'north-east': (key[0] - 1, key[1] + 1),
                'south-east': (key[0] + 1, key[1] + 1),
                'south-west': (key[0] + 1, key[1] - 1),
                'north-west': (key[0] - 1, key[1] - 1)

            }

            # Calculate Bellman Optimal State-Value Function
            for action in state_actions:
                immediate_reward = self.world.rewards[coords[action]]
                s_primes_state_vals[action] = self.env_probabilities[action] * (immediate_reward + self.gamma * self.state_values[coords[action]])
            v = list(s_primes_state_vals.values())
            k = list(s_primes_state_vals.keys())
            # REFACTOR: need to account for multiple maximums !!!
            greedy_action = k[v.index(max(v))]
            self.policy[key] = greedy_action


if __name__ == '__main__':
    world = World()
    agent = RLAgent(world, theta=0.005, gamma=0.95)
    # agent.value_iteration()
    # world.show_optimal_policy(agent.policy, title='Value Iteration')
    # world.show_rewards()
    # Policy Iteration
    # agent.policy, agent.state_values = agent.initialize()
    agent.policy_iteration()
    # print(f'agent.state_values: {agent.state_values}')
    # print(f'agent.policy: {agent.policy}')

