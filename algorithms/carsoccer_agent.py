import numpy as np


class CarSoccerAgent(object):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.chase_ball = True

    def get_actions(self, sess, state):

        if state[7] == 0:
            # We are player 1, have to score at positive x
            if abs(state[2]) < 0.3 and state[5] < 0:
                self.chase_ball = False

        else:
            # We are player 2, have to score at negative x
            if abs(state[2]) < 0.3 and state[5] > 0:
                self.chase_ball = False

        if not self.chase_ball:
            if state[7] == 0:
                if state[0] < -0.7:
                    self.chase_ball = True
            else:
                if state[0] > 0.7:
                    self.chase_ball = True

        if self.chase_ball:
            angle_corr = np.clip(state[2]*0.3, -1, 1)
            acc = 0.4 - state[6]
            return np.array([acc, angle_corr])
        else:
            angle_corr = state[4]
            acc = 0.4 - state[6]
            return np.array([acc, angle_corr])

    def train(self, sess, trajectory):
        pass