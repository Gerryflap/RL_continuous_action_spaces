import numpy as np


class SoccerAgent(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_actions(self, sess, state):
        if state[4] > 0:
            des_xs = state[4]*10
            des_ys = state[5]*10
        else:
            if abs(state[5]) > 0.2:
                des_xs = -10
                des_ys = 0
            else:
                des_xs = 0
                des_ys = 10 if state[1] < 0 else -10
        xs, ys = state[8], state[9]
        xacc, yacc = (des_xs - xs), (des_ys - ys)

        return np.array([xacc, yacc])

    def train(self, sess, trajectory):
        pass