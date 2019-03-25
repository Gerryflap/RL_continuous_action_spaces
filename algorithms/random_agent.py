import numpy as np


class RandomAgent(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_actions(self, sess, state):
        return np.random.normal(0, 1, (self.n_actions,))

    def train(self, sess, trajectory):
        pass