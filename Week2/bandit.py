import numpy as np

class Bandit:
    def __init__(self, unique_id):
        self.__mean = 0
        self.__variance = 1
        self.__unique_id = unique_id
        self.__std_deviation = np.sqrt(self.__variance)
        self.__action_value = np.random.normal(self.__mean, self.__std_deviation, 1)
        self.__reward = np.random.normal(self.__action_value, self.__std_deviation, 1)
        self.__num_times_selected = 0

    @property
    def action_value(self):
        return self.__action_value

    @property
    def reward(self):
        return self.__reward