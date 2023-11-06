import math

import numpy as np
import random
from world import World


class RLAgent:
    def __init__(self, world, theta, deterministic=True, gamma=0.95, start_state=(7,25)):
        self.world = world
        self.__theta = theta
        self.__gamma = gamma
        self.__max_iterations = 250
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

        self.__policy, self.__state_values, self.__action_probabilities = self.initialize()

    @property
    def max_iterations(self):
        return self.__max_iterations

    @property
    def action_probabilities(self):
        return self.__action_probabilities
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

        # Generate Action Probabilities
        action_probabilities = {}
        for s_idx, s in enumerate(s_vals):
            action_probabilities[s] = {}
            for action_idx, action in enumerate(policy[s]):
                action_probabilities[s][action] = 1/len(policy[s])
        return policy,s_vals,action_probabilities

    def generalized_policy_iteration(self):
        # evaluate
        V = self.state_values.copy()
        policy = self.policy.copy()
        policy_stable = False
        value_function_stable = False
        # for i in range(0, 1):
        # Evaluation
        for i in range(0, 5):
            diff = 0
            print(f'V before: {V}')
            for s_idx, s in enumerate(V):
                v = 0
                # print(f'old state value: {V[s]}')
                neighbor_coords = {
                    'north': (s[0] - 1, s[1]),
                    'east': (s[0], s[1] + 1),
                    'south': (s[0] + 1, s[1]),
                    'west': (s[0], s[1] - 1),
                    'north-east': (s[0] - 1, s[1] + 1),
                    'south-east': (s[0] + 1, s[1] + 1),
                    'south-west': (s[0] + 1, s[1] - 1),
                    'north-west': (s[0] - 1, s[1] - 1)
                }
                for action in policy[s]:
                    env_prob = self.env_probabilities[action]
                    action_prob = 1/len(policy[s])
                    reward = world.rewards[s]
                    next_state = neighbor_coords[action]
                    v += round(action_prob * env_prob*(reward + self.gamma * V[next_state]), 2)
                diff = round(abs(v - V[s]), 2)
                print(f'diff: {diff}')
                if diff < 0.05:
                    value_function_stable = True
                V[s] = v
            print(f'V after: {V}')
            print(f'value_function: {value_function_stable}')
            print(f'policy before: {policy}')
            new_policy, policy_stable = self.policy_improvement(policy, V)
            # print(f'policy after: {new_policy}')
            # print(f'new policy same?: {new_policy == policy}')
            policy = new_policy
            print(f'\n')
        self.policy = policy
            # improve


            # if policy_stable and value_function_stable:
            #     print(f'inside both case')
            #     break
            # else:
            #     print(f'inside else')
            #     print(f'policy after: {new_policy}')
            #     policy = new_policy

    def policy_iteration(self):
        '''
        Performs Policy Iteration according to Sutton, p.80, Eq.
        :return:
        '''
        state_values = self.state_values
        policy = self.policy
        count = 0
        # for i in range(0, self.max_iterations):
        #     print(f'count: {count}')
        #     count += 1
        #     print(f'old state vals: {state_values}')
        #     new_state_values = self.policy_evaluation(policy, state_values)
        #     print(f'new state vals: {new_state_values}')
        #     # print(f'new state values same than old: {new_state_values == state_values}')
        #     # print(f'old policy: {policy}')
        #     new_policy, policy_stable = self.policy_improvement(policy, new_state_values)
        #     # print(f'new policy: {new_policy}')
        #     state_values = new_state_values.copy()
        #     policy = new_policy.copy()
        #     # print(f'old policy same as new policy?: {policy == new_policy}')
        #     # print(f'policy stable: {policy_stable}')
        #     # print(f'========================================================================\n')
        #     if policy_stable:
        #         break
        # self.policy = policy
        # self.state_values = state_values
        print(f'state values before: {self.state_values}')
        self.state_values = self.policy_evaluation(self.policy, self.state_values)
        print(f'state values after: {self.state_values}')
        print(f'\n policy before imprtovement: {self.policy}')
        self.policy, policy_stable = self.policy_improvement(self.policy, self.state_values)
        print(f'\n policy after imprtovement: {self.policy}')
        # self.state_values = self.policy_evaluation(self.policy, self.state_values)
        # self.policy, policy_stable = self.policy_improvement(self.policy, self.state_values)
        # print(f'after next iteration state values: {self.state_values}')
        # print(f'after next iteration policy: {self.policy}')

    def policy_improvement(self, old_policy, state_values):
        V = state_values.copy()
        policy = old_policy.copy()
        policy_stable = True
        for s_idx, s in enumerate(V):
            available_actions = []
            old_actions = []
            if type(policy[s]) == str:
                available_actions.append(policy[s])
                old_action = [policy[s]]
            else:
                available_actions = policy[s]
                old_action = [random.choice(available_actions)]
            v = 0
            neighbor_coords = {
                'north': (s[0] - 1, s[1]),
                'east': (s[0], s[1] + 1),
                'south': (s[0] + 1, s[1]),
                'west': (s[0], s[1] - 1),
                'north-east': (s[0] - 1, s[1] + 1),
                'south-east': (s[0] + 1, s[1] + 1),
                'south-west': (s[0] + 1, s[1] - 1),
                'north-west': (s[0] - 1, s[1] - 1)
            }

            next_actions = {}
            for action in available_actions:
                next_state = neighbor_coords[action]
                immediate_reward = self.world.rewards[next_state]
                discounted_future_reward = self.gamma * state_values[next_state]
                prob = 1/len(available_actions)
                next_actions[action] = round(prob*(immediate_reward + discounted_future_reward), 2)
            # print(f'nex_actions: {next_actions}')
            val = list(next_actions.values())
            k = list(next_actions.keys())
            max_val = max(val)
            max_actions = []
            for key, value in next_actions.items():
                if value == max_val:
                    max_actions.append(key)
            greedy_action = random.choice(max_actions)
            policy[s] = [greedy_action]
            if old_action != policy[s]:
                policy_stable = False
        return policy, policy_stable


    def policy_evaluation(self, old_policy, state_values):
        V = state_values.copy()
        policy = old_policy.copy()
        for i in range(0, self.max_iterations):
            delta = 0.0
            for s_idx, s in enumerate(V):
                v = 0
                neighbor_coords = {
                    'north': (s[0] - 1, s[1]),
                    'east': (s[0], s[1] + 1),
                    'south': (s[0] + 1, s[1]),
                    'west': (s[0], s[1] - 1),
                    'north-east': (s[0] - 1, s[1] + 1),
                    'south-east': (s[0] + 1, s[1] + 1),
                    'south-west': (s[0] + 1, s[1] - 1),
                    'north-west': (s[0] - 1, s[1] - 1)
                }
                # print(f'    policy: {policy[s]}')
                action_prob = 1/len(policy[s])
                for action in policy[s]:
                    next_state = neighbor_coords[action]
                    env_prob = self.env_probabilities[action]
                    immediate_reward = world.rewards[s]
                    discounted_future_reward = self.gamma*V[next_state]
                    v += action_prob * env_prob*(immediate_reward + discounted_future_reward)

                delta = max(delta, abs(V[s] - v))
                V[s] = v
            if delta < self.theta:
                print(f'policy evaluation converged on iteration #{i}')
                break
        return V




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
            self.policy[key] = [greedy_action]


if __name__ == '__main__':
    world = World()
    agent = RLAgent(world, theta=0.005, gamma=0.95)
    # agent.value_iteration()
    # value_policy = agent.policy
    # agent.generalized_policy_iteration()
    agent.policy_iteration()
    # print(agent.policy)
    world.show_optimal_policy(agent.policy, title="Policy Iteration")
    # world.show(agent.state_values,title='policy')


