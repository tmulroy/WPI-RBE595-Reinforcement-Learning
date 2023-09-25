import numpy as np
import random
from bandit import Bandit

class EpsilonGreedy:
    def __init__(self, k, time_steps, epsilon):
        self.__num_of_bands = k
        self.__est_action_vals = np.zeros(k)
        self.__time_steps = time_steps
        self.__epsilon = epsilon
        self.__action = 0
        self.__bandits = []

        for num in range(k):
            self.__bandits = Bandit(num)
    @property
    def epsilon(self):
        return self.__epsilon

    def run(self):
        for t in range(self.__time_steps):
            rng = np.random.default_rng()
            choice = rng.choice(2, 1, p=[1-self.__epsilon, self.__epsilon])

            if t == 0:
                A = 1
            else:
                if choice == 0:
                    estimated_action_value = self.__est_action_vals[np.argmax(self.__est_action_vals)]
                    # account for multiple action-values of same value
                else:
                    random_index = random.randint(0,self.__num_of_bands-1)
                    estimated_action_value = self.__est_action_vals[random_index]
                    # account for multiple action-values of same value
