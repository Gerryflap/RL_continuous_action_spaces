"""
    A 2-player game.
    Both players have to score in the opponent's goal
"""

import math
import random
import time

import numpy as np

import pygame

screen_width = 800
screen_height = 600


def does_collide(r_xy, r_wh, r_a, c_xy, c_r):
    # IMPORTANT: rectangle width/height is interpreted in terms of a car aimed at positive X,
    #   because of this the width is over the y-axis and the height is over the x-axis!
    c_x, c_y = c_xy

    # Transform to relative
    c_x -= r_xy[0]
    c_y -= r_xy[1]

    # Rotate to align with the rectangle
    c_x, c_y = rotate(c_x, c_y, r_a)

    closest_dist_x = max(min(r_wh[1] // 2, c_x), -r_wh[1] // 2)
    closest_dist_y = max(min(r_wh[0] // 2, c_y), -r_wh[0] // 2)

    dist_x, dist_y = c_x - closest_dist_x, c_y - closest_dist_y

    return (dist_x ** 2 + dist_y ** 2) < c_r ** 2


def get_outgoing_angle(r_xy, r_wh, r_a, r_v, c_xy, c_r, c_vs):
    # IMPORTANT: rectangle width/height is interpreted in terms of a car aimed at positive X,
    #   because of this the width is over the y-axis and the height is over the x-axis!
    c_x, c_y = c_xy

    # Transform to relative
    c_x -= r_xy[0]
    c_y -= r_xy[1]

    # Rotate to align with the rectangle
    c_x, c_y = rotate(c_x, c_y, -r_a)

    # closest_dist_x = max(min(r_wh[1] / 2, c_x), -r_wh[1] / 2)
    # closest_dist_y = max(min(r_wh[0] / 2, c_y), -r_wh[0] / 2)

    closest_dist_x = r_wh[1] if c_x > 0 else -r_wh[1]
    closest_dist_y = r_wh[0] if c_y > 0 else -r_wh[0]

    dist_x, dist_y = c_x - closest_dist_x, c_y - closest_dist_y

    tvx, tvy = rotate(c_vs[0], c_vs[1], -r_a)
    if abs(closest_dist_x) < c_r:
        tvx = -tvx
        c_x = closest_dist_x + (c_r if closest_dist_x > 0 else -c_r)
        if closest_dist_x > 0 and r_v > 0:
            tvx += r_v
        if closest_dist_x < 0 and r_v < 0:
            tvx += r_v

    if abs(closest_dist_y) < c_r:
        tvy = -tvy
        c_y = closest_dist_y + (c_r if closest_dist_y > 0 else -c_r)
    c_x, c_y = rotate(c_x, c_y, r_a)
    c_x, c_y = c_x + r_xy[0], c_y + r_xy[1]

    return (c_x, c_y), rotate(tvx, tvy, r_a)


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


class Ball(object):
    def __init__(self, x, y, r, drag=1.01):
        self.x = x
        self.y = y
        self.r = r
        self.vx = 0
        self.vy = 0
        self.drag = drag

    def step(self, cars):
        self.vx /= self.drag
        self.vy /= self.drag

        if self.x < self.r:
            self.x = self.r
            self.vx = -self.vx

        if self.y < self.r:
            self.y = self.r
            self.vy = -self.vy

        if self.x > screen_width - self.r:
            self.x = screen_width - self.r
            self.vx = -self.vx

        if self.y > screen_height - self.r:
            self.y = screen_height - self.r
            self.vy = -self.vy

        # Check car collisions:
        for car in cars:
            if does_collide((car.x, car.y), car.wh, car.angle, (self.x, self.y), self.r):
                # vx, vy = rotate(car.speed, 0, car.angle)
                (self.x, self.y), (self.vx, self.vy) = get_outgoing_angle((car.x, car.y), car.wh, car.angle,
                                                                          car.speed, (self.x, self.y), self.r,
                                                                          (self.vx, self.vy))
        self.x += self.vx
        self.y += self.vy

    def pos(self):
        return int(self.x), int(self.y)


class Car(object):
    def __init__(self, x, y, angle=0, speed_limits=(-2, 10), throttle_scale=0.2, steer_scale=5e-1, color=(50, 150, 200),
                 wh=(24, 40)):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = angle
        w, h = wh
        self.wh = wh
        self.points = [(h // 2, w // 2), (h // 2, -w // 2), (-h // 2, -w // 2), (-h // 2, w // 2), (h // 2, w // 2)]
        self.color = color
        self.speed_limits = speed_limits
        self.throttle_scale = throttle_scale
        self.steer_scale = steer_scale

    def step(self, throttle, steer):
        throttle = max(-1, min(1, throttle)) * self.throttle_scale
        steer = max(-1, min(1, steer)) * self.steer_scale

        self.speed += throttle
        # Clip speed
        self.speed = max(self.speed_limits[0], min(self.speed_limits[1], self.speed))
        self.angle = (self.angle + steer) % (2 * math.pi)
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        if self.x < 0:
            self.x = 0
            self.speed = 0

        if self.y < 0:
            self.y = 0
            self.speed = 0

        if self.x > screen_width:
            self.x = screen_width
            self.speed = 0

        if self.y > screen_height:
            self.y = screen_height
            self.speed = 0

    def draw(self, surface):
        points = [rotate(x, y, self.angle) for x, y in self.points]
        points = list([(x + self.x, y + self.y) for x, y in points])
        pygame.draw.polygon(surface, self.color, points)

    def pos(self):
        return self.x, self.y


class MPCarSoccerEnv(object):
    def __init__(self, speed_limits=(-2, 10), throttle_scale=0.2, steer_scale=5e-1, max_steps=1000):
        self.car_1 = None
        self.car_2 = None
        self.ball = None
        self.steps = 0
        self.surf = None
        self.done = True
        self.speed_limits = speed_limits
        self.throttle_scale = throttle_scale
        self.steer_scale = steer_scale
        self.max_steps = max_steps

    def reset(self):

        self.car_1 = Car(0.3 * screen_width, 0.5 * screen_height, 0,
                         self.speed_limits, self.throttle_scale, self.steer_scale)

        self.car_2 = Car(0.7 * screen_width, 0.5 * screen_height, math.pi,
                         self.speed_limits, self.throttle_scale, self.steer_scale, color=(200, 50, 50))
        self.ball = Ball(screen_width / 2, screen_height / 2, 30)
        self.done = False
        self.steps = 0
        return self.__state__()

    def draw(self):
        if self.surf is None:
            pygame.init()
            self.surf = pygame.display.set_mode((screen_width, screen_height))
        self.surf.fill((50, 50, 50))
        pygame.draw.circle(self.surf, (50, 200, 50), self.ball.pos(), self.ball.r)
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
        prev_dist_1 = distance(self.car_1.pos(), self.ball.pos())
        prev_dist_2 = distance(self.car_2.pos(), self.ball.pos())
        prev_x = self.ball.x
        self.car_2.step(thrust_2, steer_2)
        self.car_1.step(thrust_1, steer_1)
        self.ball.step([self.car_1, self.car_2])
        dist_1 = distance(self.car_1.pos(), self.ball.pos())
        dist_2 = distance(self.car_2.pos(), self.ball.pos())
        dist_2_target = distance(self.ball.pos(), self.car_2.pos())

        # r_1 = 0.1 * ((prev_dist_1 - dist_1) / scale_factor - 0.001)
        # r_2 = 0.1 * ((prev_dist_2 - dist_2) / scale_factor - 0.001)

        r_1 = (self.ball.x - prev_x) / screen_width
        r_2 = -(self.ball.x - prev_x) / screen_width

        if self.steps > self.max_steps:
            self.done = True

        return self.__state__(), (r_1, r_2), self.done, None

    def __state__(self):
        c1_angle_error_target, c1_dist_target = calc_angle_error_and_dist(self.car_1, self.ball.pos())
        c1_angle_error_other, c1_dist_other = calc_angle_error_and_dist(self.car_1, self.car_2.pos())

        c2_angle_error_target, c2_dist_target = calc_angle_error_and_dist(self.car_2, self.ball.pos())
        c2_angle_error_other, c2_dist_other = calc_angle_error_and_dist(self.car_2, self.car_1.pos())

        c1_state = np.array([self.car_1.x / screen_width, self.car_1.y / screen_height, c1_angle_error_target,
                             c1_angle_error_other, c1_dist_target, c1_dist_other, self.car_1.speed * 0.1, 0])

        c2_state = np.array([self.car_2.x / screen_width, self.car_2.y / screen_height, c2_angle_error_target,
                             c2_angle_error_other, c2_dist_target, c2_dist_other, self.car_2.speed * 0.1, 1])

        return c1_state, c2_state


if __name__ == "__main__":
    env = MPCarSoccerEnv()
    env.reset()
    while True:
        _, _, d, _ = env.step(np.random.normal(0, 1, (2,)), np.random.normal(0, 1, (2,)))
        env.draw()
        if d:
            env.reset()
