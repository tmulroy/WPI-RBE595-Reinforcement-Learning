import numpy as np
import matplotlib.pyplot as plt


class World:
    def __init__(self, goal=(7,10)):
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

    @state_values.setter
    def state_values(self, new_state_values):
        self.__state_values = new_state_values

    def show(self, values):
        fig, ax = plt.subplots()
        ax.set_title('World')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        im = ax.imshow(values)
        plt.show(block=False)

    def convert_dict_to_arr(self, x):
        # for key,value in x:
        pass


    # def show_values(self):

    def show_rewards_arr(self):
        fig, ax = plt.subplots()
        ax.set_title('Rewards')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        im = ax.imshow(self.rewards_arr)
        it = np.nditer(self.rewards_arr, flags=['multi_index'])
        for reward in it:
            text = ax.text(it.multi_index[1], it.multi_index[0], reward, color='w', fontsize='xx-small')
        plt.show(block=False)

