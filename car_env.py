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


scale_factor = distance((0, 0), (screen_width, screen_height))


class Car(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.points = [(8, 5), (8, -5), (-8, -5), (-8, 5), (8, 5)]

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
        pygame.draw.polygon(surface, (50, 150, 200), points)

    def pos(self):
        return self.x, self.y


class CarEnv(object):
    def __init__(self, use_easy_state):
        self.car = None
        self.target = None
        self.steps = 0
        self.surf = None
        self.done = True
        self.use_easy_state = use_easy_state

    def reset(self):
        self.car = Car(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1))
        self.target = random.randint(0, screen_width - 1), random.randint(0, screen_height - 1)
        self.done = False
        self.steps = 0
        return self.__state__()

    def draw(self):
        if self.surf is None:
            pygame.init()
            self.surf = pygame.display.set_mode((screen_width, screen_height))
        self.surf.fill((50, 50, 50))
        pygame.draw.circle(self.surf, (200, 50, 50), self.target, 30)
        self.car.draw(self.surf)
        pygame.display.flip()

    def step(self, action):
        self.steps += 1
        if self.done:
            raise RuntimeWarning("Calling step on environment that is currently in the 'done' state!")
        thrust, steer = action
        prev_dist = distance(self.car.pos(), self.target)
        self.car.step(thrust, steer)
        dist = distance(self.car.pos(), self.target)
        r = (prev_dist - dist)/scale_factor - 0.001

        if not is_in_bounds(self.car.x, self.car.y) or self.steps > 1000:
            r -= 10
            self.done = True

        if dist < 30:
            r += 10
            self.done = True
        return self.__state__(), r, self.done, None

    def __state__(self):
        if not self.use_easy_state:
            return np.array([self.car.x/screen_width, self.car.y/screen_height, self.car.angle/(2*math.pi),
                             self.car.speed*0.1, self.target[0]/screen_width, self.target[1]/screen_height])
        else:
            angle_to_target = math.atan2(self.car.y - self.target[1], self.car.x - self.target[0])
            distance_to_target = distance(self.car.pos(), self.target)
            distance_to_target /= scale_factor
            angle_error = (self.car.angle - angle_to_target)%(2*math.pi)
            angle_error -= math.pi
            angle_error /= math.pi

            return np.array([self.car.x / screen_width, self.car.y / screen_height, angle_error,
                             distance_to_target, self.car.angle/(2*math.pi), self.car.speed * 0.1])



if __name__ == "__main__":
    env = CarEnv(False)
    target_speed = 8

    while True:
        state = env.reset()
        done = False
        while not done:
            speed = state[3]
            throttle = target_speed - speed
            state, r, done, _ = env.step((throttle, 2 * random.random() - 1))
            print(r, speed)
            env.draw()
            time.sleep(1/60)
