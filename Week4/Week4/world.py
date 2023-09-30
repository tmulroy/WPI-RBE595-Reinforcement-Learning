import numpy as np
import matplotlib.pyplot as plt


class World:
    def __init__(self, goal=(7, 10)):
        self.__grid = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

        self.__goal = goal
        self.__grid_dict = self.convert_ndarray_to_dict(self.__grid)
        # Rewards as a numpy.ndarray()
        rewards_arr = np.array([-1.0 if state == 0 else -50.0 for row in self.__grid for state in row])
        rewards_arr = np.reshape(rewards_arr, (15,51))
        rewards_arr[self.__goal[0], self.__goal[1]] = 100.00
        self.__rewards_arr = rewards_arr
        self.__state_values = {}
        # Generate Rewards as a dict {}
        rewards = {}
        for y in range(0,self.__grid.shape[0]):
            for x in range(0,self.__grid.shape[1]):
                # print(f'(y,x): {y,x}')
                rewards[(y,x)] = self.__grid[y,x]
                if self.__grid[y,x] == 1:
                    rewards[(y, x)] = -50.0
                elif self.__grid[y,x] == 0:
                    rewards[(y,x)] = -1.0
                else:
                    pass
        rewards[self.__goal] = 100.0

        self.__rewards = rewards

        # grid: {
        #   (0,1): {state_value: 0, reward: 1}
        # }

    @property
    def rewards(self):
        return self.__rewards

    @property
    def rewards_arr(self):
        return self.__rewards_arr

    @property
    def grid(self):
        return self.__grid

    @property
    def state_values(self):
        return self.__state_values

    @property
    def grid_dict(self):
        return self.__grid_dict

    @state_values.setter
    def state_values(self, new_state_values):
        self.__state_values = new_state_values

    def show(self, values, title):
        if type(values) == dict:
            values = self.convert_dict_to_arr(values)
        fig,ax = plt.subplots()
        ax.set_title(title)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        im = ax.imshow(values)
        plt.show()

    def convert_dict_to_arr(self, x):
        values_arr = np.empty((self.grid.shape[0], self.grid.shape[1]))
        for key in x.keys():
            values_arr[key[0], key[1]] = x[key]
        return values_arr

    def convert_ndarray_to_dict(self, arr):
        d = {}
        for y in range(0, arr.shape[0]):
            for x in range(0, arr.shape[1]):
                d[(y,x)] = arr[y,x]
        return d


    def show_optimal_policy(self, policy, title):
        # REFACTOR: don't plot policy for obstacle
        fig, ax = plt.subplots()
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        im = ax.imshow(self.convert_dict_to_arr(self.rewards), cmap='Greys')
        ax.set_title(f'{title} Optimal Policy')
        for state, action in policy.items():
            action = action[0]
            if self.grid_dict[state] == 0:
                if action == 'north':
                    plt.arrow(state[1], state[0], 0, -0.35, head_width=0.1)
                elif action == 'east':
                    plt.arrow(state[1], state[0], 0.35, 0, head_width=0.1)
                elif action == 'south':
                    plt.arrow(state[1], state[0], 0, 0.35, head_width=0.1)
                elif action == 'west':
                    plt.arrow(state[1], state[0], -0.35, 0, head_width=0.1)
                elif action == 'north-east':
                    plt.arrow(state[1], state[0], 0.35, -0.35, head_width=0.1)
                elif action == 'north-west':
                    plt.arrow(state[1], state[0], -0.35, -0.35, head_width=0.1)
                elif action == 'south-east':
                    plt.arrow(state[1], state[0], 0.35, 0.35, head_width=0.1)
                elif action == 'south-west':
                    plt.arrow(state[1], state[0], -0.35, 0.35, head_width=0.1)
                else:
                    plt.arrow(state[1], state[0], 0, 0.35, head_width=0.1)
        plt.show()

    def show_rewards(self):
        # rewards = self.convert_dict_to_arr(self.rewards)
        fig, ax = plt.subplots()
        ax.set_title('Rewards')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        im = ax.imshow(self.convert_dict_to_arr(self.rewards))
        for key in self.rewards.keys():
            text = ax.text(key[1], key[0], self.rewards[key], color='w', fontsize='xx-small')
        plt.show()

