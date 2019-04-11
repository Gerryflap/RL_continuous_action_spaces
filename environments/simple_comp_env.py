"""
    Simple competition env.
    2 agents need to move their x-pos to 0, the first to get within a given range of 0 wins
    if the agent deviates too far of the center it will lose
"""
import random
import numpy as np


class SimpleCompEnv(object):
    max_steps = 30
    acceptance_range = 0.01
    acceleration_multiplier = 0.1

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.agent_1_pos = random.random() * 2 - 1
        self.agent_2_pos = self.agent_1_pos
        return self._get_states()

    def _get_states(self):
        return np.array([self.agent_1_pos, self.agent_2_pos]), np.array([self.agent_2_pos, self.agent_1_pos])

    def step(self, a_p1, a_p2):
        self.agent_1_pos += a_p1[0] * self.acceleration_multiplier
        self.agent_2_pos += a_p2[0] * self.acceleration_multiplier

        if self.max_steps < self.steps:
            return self._get_states(), (-1, -1), True, None
        self.steps += 1

        if abs(self.agent_1_pos) > 1 and abs(self.agent_2_pos) > 1:
            return self._get_states(), (-1, -1), True, None

        if abs(self.agent_1_pos) > 1:
            return self._get_states(), (-1, 1), True, None

        if abs(self.agent_2_pos) > 1:
            return self._get_states(), (1, -1), True, None

        if abs(self.agent_1_pos) < self.acceptance_range and abs(self.agent_2_pos) < self.acceptance_range:
            return self._get_states(), (1, 1), True, None

        if abs(self.agent_1_pos) < self.acceptance_range:
            return self._get_states(), (1, -1), True, None

        if abs(self.agent_2_pos) < self.acceptance_range:
            return self._get_states(), (-1, 1), True, None

        return self._get_states(), (0, 0), False, None

    def render(self):
        print("Step: ")
        print("%.3f \t%.3f\n"%(self.agent_1_pos, self.agent_2_pos))

