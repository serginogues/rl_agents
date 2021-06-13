import numpy as np
from gym.spaces import Discrete


class BitFlipEnv:
    """
    A simple bit flip environment
    Bit of the current state flips as an action
    Reward of -1 for each step
    """
    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)
        self.action_space = Discrete(len(self.state))

    def render(self):
        print("State: ", self.state)
        print("Goal: ", self.goal)

    def reset(self):
        """
        Resets the environment with new state and goal
        """
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)
        return self.state

    def step(self, action):
        """
        Returns updated_state, reward, and done for the step taken
        """
        self.state[action] = self.state[action] ^ 1
        done = False
        if np.array_equal(self.state, self.goal):
            done = True
            reward = 0
        else:
            reward = -1
        return np.copy(self.state), reward, done, {}

    def print_state(self):
        """
        Prints the current state
        """
        print('Current State:', self.state)

    def evaluate(self, action, state, goal):
        new_reward = -1
        new_done = False
        state[action] = state[action] ^ 1
        if np.array_equal(state, goal):
            new_done = True
            new_reward = 0
        return new_reward, new_done
