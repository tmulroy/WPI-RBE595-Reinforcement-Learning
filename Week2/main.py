import random
import matplotlib.pyplot as plt
import numpy as np
import time


def run(epsilon, time_steps):
    k = 10 # number of bandits
    mean = 0
    variance = 1
    std_deviation = np.sqrt(variance)
    distr_size = 10
    rewards = np.empty(time_steps)

    # Initialize estimated action-values (Q), actual action-values (q) and action counts (N)
    Q = np.zeros(k)
    q = np.random.normal(mean, std_deviation, distr_size)
    N = np.zeros(k)

    # Generate actual rewards based on Gaussian Distribution
    R = np.empty(k)
    for i in range(k):
        R[i] = np.random.normal(q[i], std_deviation)

    # Run
    for t in range(time_steps):
        rng = np.random.default_rng()
        choice = rng.choice(2, 1, p=[1-epsilon, epsilon])
        '''
        Choose argmax(Q(a)) with probability 1-epsilon
        and random(Q(a)) with probability epsilon
        '''
        if t == 0:
            A = 1
        else:
            if choice == 0:
                est = Q[np.argmax(Q)]
                A = np.where(Q == est)[0]
                if len(A) > 1:
                    A = A[0]
            else:
                random_index = random.randint(0, 9)
                est = Q[random_index]

                A = np.where(Q == est)[0]
                if len(A) > 1:
                    A = A[0]

        # Reward for selected action A
        reward = q[A]

        # Update number of times bandit was selected
        N[A] += 1

        # Incremental Sample Average Update
        Q_next = Q[A] + (1/N[A]) * (reward - Q[A])
        Q[A] = Q_next

        # Keep record of reward for each time step
        rewards[t] = reward

    return rewards


if __name__ == '__main__':

    epsilon_greedy = []
    epsilon_mid = []
    epsilon_low = []

    start = time.time()
    for i in range(2000):
        r1 = run(epsilon=0.01, time_steps=1000)
        r2 = run(epsilon=0.1, time_steps=1000)
        r3 = run(epsilon=0, time_steps=1000)
        epsilon_greedy.append(r3)
        epsilon_mid.append(r2)
        epsilon_low.append(r1)
    end = time.time()
    print(f'runtime: {end - start}')


    avg_epsilon_greedy = np.mean(epsilon_greedy, axis=0)
    avg_epsilon_mid = np.mean(epsilon_mid, axis=0)
    avg_epsilon_low = np.mean(epsilon_low, axis=0)

    x = np.linspace(0, 1000,1000)
    fig, ax = plt.subplots()
    ax.plot(x, avg_epsilon_greedy, label='epsilon = 0')
    ax.plot(x, avg_epsilon_low, label='epsilon = 0.01')
    ax.plot(x, avg_epsilon_mid, label='epsilon = 0.1')
    ax.legend()
    plt.show()