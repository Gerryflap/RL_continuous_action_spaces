"""
    A 2-player game.
    The blue car has to reach the Green target circle to win without leaving the screen.
    The red car has to tag the blue car to win, but is not allowed to enter the target circle (this costs points).
"""

import math
import random
import time

import numpy as np

import pygame

screen_width = 800
screen_height = 800


def set_random_seed(seed):
    random.seed(seed)


def rotate(x, y, angle):
    new_x = math.cos(angle) * x - math.sin(angle) * y
    new_y = math.sin(angle) * x + math.cos(angle) * y
    return new_x, new_y


def distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def is_in_bounds(x, y):
    return 0 <= x <= screen_width and 0 <= y <= screen_height


def calc_angle_error_and_dist(car, pos):
    angle_to_target = math.atan2(car.y - pos[1], car.x - pos[0])
    distance_to_target = distance(car.pos(), pos)
    distance_to_target /= scale_factor
    angle_error = (car.angle - angle_to_target) % (2 * math.pi)
    angle_error -= math.pi
    angle_error /= math.pi
    return angle_error, distance_to_target


scale_factor = distance((0, 0), (screen_width, screen_height))


class Car(object):
    def __init__(self, x, y, color=(50, 150, 200)):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.points = [(8, 5), (8, -5), (-8, -5), (-8, 5), (8, 5)]
        self.color = color

    def step(self, throttle, steer):
        throttle = max(-1, min(1, throttle)) * 0.2
        steer = max(-1, min(1, steer)) * 5e-1

        self.speed += throttle
        # Clip speed
        self.speed = max(-2, min(10, self.speed))
        self.angle = (self.angle + steer) % (2 * math.pi)
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

    def draw(self, surface):
        points = [rotate(x, y, self.angle) for x, y in self.points]
        points = list([(x + self.x, y + self.y) for x, y in points])
        pygame.draw.polygon(surface, self.color, points)

    def pos(self):
        return self.x, self.y


class MPCarEnv(object):
    def __init__(self):
        self.car_1 = None
        self.car_2 = None
        self.target = None
        self.steps = 0
        self.surf = None
        self.done = True

    def reset(self):
        self.target = random.randint(0, screen_width - 1), random.randint(0, screen_height - 1)

        # Spawn on opposing side:
        # self.car_1 = Car(screen_width - 1 - self.target[0], screen_height - 1 - self.target[1])
        self.car_1 = Car(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1))
        self.car_2 = Car(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1), color=(200, 50, 50))
        self.done = False
        self.steps = 0
        return self.__state__()

    def draw(self):
        if self.surf is None:
            pygame.init()
            self.surf = pygame.display.set_mode((screen_width, screen_height))
        self.surf.fill((50, 50, 50))
        pygame.draw.circle(self.surf, (50, 200, 50), self.target, 30)
        pygame.draw.circle(self.surf, (100, 50, 50), (int(self.car_2.x), int(self.car_2.y)), 20, 3)
        self.car_1.draw(self.surf)
        self.car_2.draw(self.surf)
        pygame.display.flip()

    def step(self, action_1, action_2):
        self.steps += 1
        if self.done:
            raise RuntimeWarning("Calling step on environment that is currently in the 'done' state!")
        thrust_1, steer_1 = action_1
        thrust_2, steer_2 = action_2
        prev_dist_1 = distance(self.car_1.pos(), self.target)
        prev_dist_2 = distance(self.car_1.pos(), self.car_2.pos())
        self.car_2.step(thrust_2, steer_2)
        self.car_1.step(thrust_1, steer_1)
        dist_1 = distance(self.car_1.pos(), self.target)
        dist_2 = distance(self.car_1.pos(), self.car_2.pos())
        dist_2_target = distance(self.target, self.car_2.pos())

        r_1 = (prev_dist_1 - dist_1) / scale_factor - 0.001
        r_2 = (prev_dist_2 - dist_2) / scale_factor - 0.001

        if not is_in_bounds(self.car_1.x, self.car_1.y) or self.steps > 1000:
            r_1 -= 1
            self.done = True

        if dist_1 < 30:
            r_1 += 1
            r_2 -= 1
            self.done = True

        if dist_2 < 20:
            r_1 -= 1
            r_2 += 1
            self.done = True

        if dist_2_target < 30:
            r_2 -= 0.1

        return self.__state__(), (r_1, r_2), self.done, None

    def __state__(self):
        c1_angle_error_target, c1_dist_target = calc_angle_error_and_dist(self.car_1, self.target)
        c1_angle_error_other, c1_dist_other = calc_angle_error_and_dist(self.car_1, self.car_2.pos())

        c2_angle_error_target, c2_dist_target = calc_angle_error_and_dist(self.car_2, self.target)
        c2_angle_error_other, c2_dist_other = calc_angle_error_and_dist(self.car_2, self.car_1.pos())

        c1_state = np.array([self.car_1.x / screen_width, self.car_1.y / screen_height, c1_angle_error_target,
                             c1_angle_error_other, c1_dist_target, c1_dist_other, self.car_1.speed * 0.1])

        c2_state = np.array([self.car_2.x / screen_width, self.car_2.y / screen_height, c2_angle_error_target,
                             c2_angle_error_other, c2_dist_target, c2_dist_other, self.car_2.speed * 0.1])

        return c1_state, c2_state
