import numpy as np


class PrisonersDilemmaEnv(object):
    final_state = np.array([0, 1], dtype=np.float32)

    def reset(self):
        s = np.array([1, 0], dtype=np.float32)
        return (s, s)

    def step(self, a_p1, a_p2):

        if a_p1 < 0 and a_p2 < 0:
            r = (-0.5, -0.5)
        elif a_p1 < 0 <= a_p2:
            r = (-2, 0)
        elif a_p1 >= 0 > a_p2:
            r = (0, -2)
        else:
            r = (-2, -2)

        return (self.final_state, self.final_state), r, True, None

    def draw(self):
        pass

    def render(self):
        pass
