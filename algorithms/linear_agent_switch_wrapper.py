import random

import numpy as np


class LinearAgentSwitchWrapper(object):
    def __init__(self, learning_agent, initial_agent, n_updates):
        self.learning_agent = learning_agent
        self.initial_agent = initial_agent
        self.n_updates = n_updates
        self.updates_count = 0
        self.p = 0

    def get_actions(self, sess, state):
        if random.random() > self.p:
            return self.initial_agent.get_actions(sess, state)
        else:
            return self.learning_agent.get_actions(sess, state)

    def train(self, sess, trajectory):
        self.updates_count += 1
        self.p = min(self.updates_count/self.n_updates, 1)
        self.initial_agent.train(sess, trajectory)
        self.learning_agent.train(sess, trajectory)
