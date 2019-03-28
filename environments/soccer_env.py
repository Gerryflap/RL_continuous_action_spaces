import numpy as np
import pygame
import random


def text_to_screen(screen, text, x, y, size = 10,
            color = (200, 200, 200), font_type = 'Cantarell'):
    try:

        text = str(text)
        font = pygame.font.SysFont(font_type, size)
        text = font.render(text, True, color)
        screen.blit(text, (x, y))

    except Exception as e:
        print('Font Error, saw it coming')
        raise e


class SoccerEnvironment(object):
    action_space = 2
    width = 600
    height = 300
    circle_radius = 25
    # resistance_factors = [1.02, 1.02, 1.01]
    resistance_factors = [1.01, 1.01, 1.02]
    acceleration = 0.5
    bounce_resistance = [1.3, 1.3, 2]
    max_steps_after_bounce = 400
    max_steps = 5000
    object_masses = [1, 1, 0.3]

    def __init__(self, gui=True, add_random=True):
        if gui:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = None
        self.add_random = add_random
        self.reset()


    def reset(self):
        self.p1_pos = (0.1 * self.width, 0.5 * self.height)
        self.p2_pos = (0.9 * self.width, 0.5 * self.height)
        self.ball_pos = (0.5 * self.width, 0.5 * self.height)

        self.p1_speed = (0, 0)
        self.p2_speed = (0, 0)
        self.ball_speed = (0, 0) #(10 * (random.random()*2-1), 10 * (random.random()*2-1))
        self.steps = 0
        self.steps_since_ball_touch = 0
        return self.__gen_state__()

    def step(self, p1_action, p2_action):
        # Should return state, reward, done, info
        self.p1_speed = self.__do_action__(self.p1_speed, p1_action, 1)
        self.p2_speed = self.__do_action__(self.p2_speed, p2_action, 2)
        state = [
            (self.p1_pos, self.p1_speed),
            (self.p2_pos, self.p2_speed),
            (self.ball_pos, self.ball_speed)
        ]
        new_state, cols = self.__run_physics__(state)
        rewards, done = self.__check_goal__(cols)
        self.p1_pos, self.p1_speed = new_state[0]
        self.p2_pos, self.p2_speed = new_state[1]
        self.ball_pos, self.ball_speed = new_state[2]
        self.steps += 1
        self.steps_since_ball_touch += 1

        return self.__gen_state__(), rewards, done, None

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (0, 255, 0), (0, 0), (0, 300), 10)
        pygame.draw.line(self.screen, (0, 255, 0), (self.width, 0), (self.width, 300), 10)
        x, y = self.p1_pos
        x, y = int(x), int(y)
        pygame.draw.circle(self.screen, (255, 0, 0), (x, y), self.circle_radius)
        x, y = self.p2_pos
        x, y = int(x), int(y)
        pygame.draw.circle(self.screen, (0, 0, 255), (x, y), self.circle_radius)
        x, y = self.ball_pos
        x, y = int(x), int(y)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), self.circle_radius)
        s = ""
        for i, num in enumerate(self.__gen_state__()[0]):
            s += "%2.2f " % num
            if i % 2 == 1:
                s += "|| "
        text_to_screen(self.screen, s, 10, 250)
        pygame.display.flip()

    def __gen_state__(self):
        w = self.width
        h = self.height
        hw = w / 2
        hh = h / 2
        p1_state = np.array([
            (self.p1_pos[0] - hw) / hw, (self.p1_pos[1] - hh) / hh,
            (self.p2_pos[0] - self.p1_pos[0]) / hw, (self.p2_pos[1] - self.p1_pos[1])/ hh,
            (self.ball_pos[0] - self.p1_pos[0]) / hw, (self.ball_pos[1] - self.p1_pos[1]) / hh,
            self.p1_speed[0], self.p1_speed[1],
            self.p2_speed[0], self.p2_speed[1],
            self.ball_speed[0], self.ball_speed[1],
        ])
        p2_state = np.array([
            -(self.p2_pos[0] - hw) / hw, (self.p2_pos[1] - hh) / hh,
            -(self.p1_pos[0] - self.p2_pos[0]) / hw, (self.p1_pos[1] - self.p2_pos[1])/ hh,
            -(self.ball_pos[0] - self.p2_pos[0]) / hw, (self.ball_pos[1] - self.p2_pos[1]) / hh,
            -self.p2_speed[0], self.p2_speed[1],
            -self.p1_speed[0], self.p1_speed[1],
            -self.ball_speed[0], self.ball_speed[1],
        ])
        if self.add_random:
            p1_state = np.concatenate((p1_state, np.random.normal(0, 1, (1,))), axis=0)
            p2_state = np.concatenate((p2_state, np.random.normal(0, 1, (1,))), axis=0)
        return p1_state, p2_state

    def __run_physics__(self, objects):
        outputs = objects[:]
        for i, o in enumerate(objects):
            pos, speed = o
            outputs[i] = (
                (pos[0] + speed[0], pos[1] + speed[1]),
                (speed[0] / self.resistance_factors[i], speed[1] / self.resistance_factors[i])
            )
        for i, o in enumerate(outputs):
            pos, speed = o
            x, y = pos
            xs, ys = speed
            if x > self.width - self.circle_radius and not (0 < y < 300):
                x = self.width - self.circle_radius
                xs = -xs/self.bounce_resistance[i]
            if x < self.circle_radius and not (0 < y < 300):
                x = self.circle_radius
                xs = -xs/self.bounce_resistance[i]
            if x > self.width + self.circle_radius:
                x = self.width + self.circle_radius
                xs = -xs/self.bounce_resistance[i]
            if x < -self.circle_radius:
                x = -self.circle_radius
                xs = -xs/self.bounce_resistance[i]
            if y > self.height - self.circle_radius:
                y = self.height - self.circle_radius
                ys = -ys/self.bounce_resistance[i]
            if y < self.circle_radius:
                y = self.circle_radius
                ys = -ys/self.bounce_resistance[i]
            outputs[i] = ((x, y), (xs, ys))

        min_sq_dist = (self.circle_radius * 2) ** 2
        update_queue = []
        collisions = set()
        for i, o in enumerate(outputs):
            pos, speed = o
            for j, o2 in enumerate(outputs):
                pos2, speed2 = o2
                if i == j:
                    continue
                sq_dist = (pos[0] - pos2[0]) ** 2 + (pos[1] - pos2[1]) ** 2
                if sq_dist == 0:
                    continue
                if sq_dist < min_sq_dist:
                    xs, ys = speed2
                    x, y = pos2[0] - pos[0], pos2[1] - pos[1]
                    total_s = (xs ** 2 + ys ** 2) ** 0.5
                    total_s *= self.object_masses[j]/self.object_masses[i]

                    dist = sq_dist ** 0.5
                    xn = x / dist
                    yn = y / dist

                    new_v = -xn * total_s, -yn * total_s
                    update_queue.append((i, new_v))
                    collisions.add((i, j))
        for i, vs in update_queue:
            pos, speed = outputs[i]
            xs, ys = speed
            xsd, ysd = vs
            xsd /= self.bounce_resistance[i]
            ysd /= self.bounce_resistance[i]
            outputs[i] = pos, (xs + xsd, ys + ysd)
        return outputs, collisions

    def __do_action__(self, speed, action, player):
        xs, ys = speed
        xacc, yacc = np.clip(action, -1, 1) * self.acceleration
        if player == 2:
            xacc = -xacc
        xs += xacc
        ys += yacc

        # if action == 0:
        #     return xs + self.acceleration, ys
        # if action == 1:
        #     return xs, ys + self.acceleration
        # if action == 2:
        #     return xs - self.acceleration, ys
        # if action == 3:
        #     return xs, ys - self.acceleration
        return xs, ys

    def __check_goal__(self, collisions):
        x, y = self.ball_pos
        if x > self.width:
            self.reset()
            return (1.0, -1.0), True
        x, y = self.ball_pos
        if x < 0:
            self.reset()
            return (-1.0, 1.0), True
        if self.steps > self.max_steps or self.steps_since_ball_touch > self.max_steps_after_bounce:
            self.reset()
            return (-1.0, -1.0), True
        r1 = 0
        r2 = 0
        if self.get_total_speed(self.p1_speed) < 0.1:
            r1 += 0
        if self.get_total_speed(self.p2_speed) < 0.1:
            r2 += 0

        if (0, 2) in collisions:
            r1 += 0
            self.steps_since_ball_touch = 0
        if (1, 2) in collisions:
            r2 += 0
            self.steps_since_ball_touch = 0

        return (r1, r2), False

    def action_space_sample(self):
        return random.randint(0, 4)

    def get_total_speed(self, speed_tuple):
        return (speed_tuple[0]**2 + speed_tuple[1]**2)**0.5


if __name__ == "__main__":

    import time

    env = SoccerEnvironment()
    while 1:
        env.step(np.random.normal(0, 1, (2,)), np.random.normal(0, 1, (2,)))
        env.render()
        time.sleep(1 / 60)
