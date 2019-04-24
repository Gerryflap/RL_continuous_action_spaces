import math
import time
import numpy as np

import Box2D
from gym.envs.classic_control import rendering

field_width = 150  # 100
field_height = field_width * 3 / 5  # 60

screen_width = 500
screen_height = 300


def transform_pos(pos):
    x, y = pos
    return int(screen_width * ((x / field_width) + 0.5)), int((0.5 - (y / field_height)) * screen_height)


def rotate(x, y, angle):
    new_x = math.cos(angle) * x - math.sin(angle) * y
    new_y = math.sin(angle) * x + math.cos(angle) * y
    return new_x, new_y


def velocity(obj, return_vector=False):
    vels = obj.__GetLinearVelocity()
    if return_vector:
        return vels
    else:
        return (vels[0] ** 2 + vels[1] ** 2) ** 0.5


def distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def apply_wheel_resistance(car):
    currentRightNormal = car.GetWorldVector(Box2D.b2Vec2(0, 1))
    latVel = Box2D.b2Dot(currentRightNormal, car.__GetLinearVelocity()) * currentRightNormal
    impulse = car.__GetMass() * -latVel
    car.ApplyLinearImpulse(impulse, car.GetWorldPoint((0, 0)), True)


scale_factor = distance((0, 0), (field_width, field_height))


def calc_angle_error_and_dist(car, pos):
    angle_to_target = math.atan2(car.position[1] - pos[1], car.position[0] - pos[0])
    distance_to_target = distance(car.position, pos)
    distance_to_target /= scale_factor
    angle_error = (car.angle - angle_to_target) % (2 * math.pi)
    angle_error -= math.pi
    angle_error /= math.pi
    return angle_error, distance_to_target

    # x, y = transform_pos(ground.position)
    # pygame.draw.rect(display, (150, 150, 150), pygame.Rect(x, y - 5*10, 10*10, 5*10))

    # car.ApplyAngularImpulse(200, True)

    # car.ApplyLinearImpulse(rotate(70, 0, car.angle), car.GetWorldPoint((0, 0)), True)

    # car.ApplyAngularImpulse(-200, True)


class CarSoccerEnv(object):
    def __init__(self, max_steps=-1, distance_to_ball_r_factor=0.0, distance_to_goal_r_factor=0.0):
        self.w = None
        self.ball = None
        self.car_1 = None
        self.car_2 = None
        self.viewer = None
        self.max_steps = max_steps
        self.steps = 0
        self.distance_to_ball_r_factor = distance_to_ball_r_factor
        self.distance_to_goal_r_factor = distance_to_goal_r_factor

    def reset(self):
        self.steps = 0
        self.w = Box2D.b2World(gravity=(0, 0))

        # Construct ground
        ground_top = self.w.CreateStaticBody(position=(field_width / 2, field_height / 2))
        ground_top.CreateFixture(shape=Box2D.b2.polygonShape(box=(field_width, 5)), density=1.0, restitution=0.0,
                                 friction=1.0)

        ground_bot = self.w.CreateStaticBody(position=(field_width / 2, -field_height / 2))
        ground_bot.CreateFixture(shape=Box2D.b2.polygonShape(box=(field_width, 5)), density=1.0, restitution=0.0,
                                 friction=1.0)

        # ground_left_top = self.w.CreateStaticBody(position=(-field_width / 2, -field_height / 2 + 5 * field_height/6))
        # ground_left_top.CreateFixture(shape=Box2D.b2.polygonShape(box=(5, field_height/6)), density=1.0, restitution=0.0,
        #                           friction=1.0)
        #
        # ground_right_top = self.w.CreateStaticBody(position=(field_width / 2, -field_height / 2 + 5*field_height/6))
        # ground_right_top.CreateFixture(shape=Box2D.b2.polygonShape(box=(5, field_height/6)), density=1.0, restitution=0.0,
        #                            friction=1.0)
        #
        # ground_left_bot = self.w.CreateStaticBody(position=(-field_width / 2, -field_height / 2))
        # ground_left_bot.CreateFixture(shape=Box2D.b2.polygonShape(box=(5, field_height/3)), density=1.0, restitution=0.0,
        #                           friction=1.0)
        #
        # ground_right_bot = self.w.CreateStaticBody(position=(field_width / 2, -field_height / 2))
        # ground_right_bot.CreateFixture(shape=Box2D.b2.polygonShape(box=(5, field_height/3)), density=1.0, restitution=0.0,
        #                            friction=1.0)

        self.ball = self.w.CreateDynamicBody(position=(0, 0), bullet=True, linearDamping=0.3)
        self.ball.CreateCircleFixture(radius=3, density=0.1, restitution=0.5, friction=0.1)

        self.car_1 = self.w.CreateDynamicBody(position=(-10, 0), bullet=True, linearDamping=0.7, angularDamping=10.0)
        self.car_1.CreateFixture(shape=Box2D.b2.polygonShape(box=(2.5, 1.8)), density=5.0, restitution=0.5,
                                 friction=1.0)

        self.car_2 = self.w.CreateDynamicBody(position=(10, 0), bullet=True, linearDamping=0.7, angularDamping=10.0, angle=math.pi)
        self.car_2.CreateFixture(shape=Box2D.b2.polygonShape(box=(2.5, 1.8)), density=5.0, restitution=0.5,
                                 friction=1.0)

        return self.__state__()

    def step(self, action_1, action_2):

        self.car_1.ApplyLinearImpulse(rotate(action_1[0] * 170, 0, self.car_1.angle), self.car_1.GetWorldPoint((0, 0)),
                                      True)
        self.car_1.ApplyAngularImpulse(action_1[1] * 200, True)

        self.car_2.ApplyLinearImpulse(rotate(action_2[0] * 170, 0, self.car_2.angle), self.car_2.GetWorldPoint((0, 0)),
                                      True)
        self.car_2.ApplyAngularImpulse(action_2[1] * 200, True)
        # print(self.car_1.angle, calc_angle_error_and_dist(self.car_1, self.ball.position)[0])

        p1_old_ball_dist = distance(self.car_1.position, self.ball.position)
        p2_old_ball_dist = distance(self.car_2.position, self.ball.position)

        p1_old_goal_dist = distance((field_width / 2, 0), self.ball.position)
        p2_old_goal_dist = distance((-field_width / 2, 0), self.ball.position)

        # Car counter-force
        apply_wheel_resistance(self.car_1)
        apply_wheel_resistance(self.car_2)

        self.w.Step(1 / 30, 3, 3)
        self.w.ClearForces()

        p1_ball_dist = distance(self.car_1.position, self.ball.position)
        p2_ball_dist = distance(self.car_2.position, self.ball.position)
        p1_goal_dist = distance((field_width / 2, 0), self.ball.position)
        p2_goal_dist = distance((-field_width / 2, 0), self.ball.position)

        r_1, r_2 = 0, 0

        r_1 += self.distance_to_ball_r_factor * (p1_old_ball_dist - p1_ball_dist) / scale_factor
        r_2 += self.distance_to_ball_r_factor * (p2_old_ball_dist - p2_ball_dist) / scale_factor

        r_1 += self.distance_to_goal_r_factor * (p1_old_goal_dist - p1_goal_dist) / scale_factor
        r_2 += self.distance_to_goal_r_factor * (p2_old_goal_dist - p2_goal_dist) / scale_factor

        if self.ball.position[0] > field_width / 2:
            r_1 += 1
            r_2 -= 1
            return self.__state__(), (r_1, r_2), True, None

        if self.ball.position[0] < -field_width / 2:
            r_1 -= 1
            r_2 += 1
            return self.__state__(), (r_1, r_2), True, None

        if self.car_1.position[0] < -field_width / 2:
            self.car_1.ApplyLinearImpulse((1000, 0),
                                          self.car_1.GetWorldPoint((0, 0)),
                                          True)

        if self.car_1.position[0] > field_width / 2:
            self.car_1.ApplyLinearImpulse((-1000, 0),
                                          self.car_1.GetWorldPoint((0, 0)),
                                          True)

        if self.car_2.position[0] < -field_width / 2:
            self.car_2.ApplyLinearImpulse((1000, 0),
                                          self.car_2.GetWorldPoint((0, 0)), True)

        if self.car_2.position[0] > field_width / 2:
            self.car_2.ApplyLinearImpulse((-1000, 0),
                                          self.car_2.GetWorldPoint((0, 0)), True)

        self.steps += 1
        if self.max_steps != -1 and self.steps > self.max_steps:
            return self.__state__(), (-1, -1), True, None

        return self.__state__(), (r_1, r_2), False, None

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-field_width / 2, field_width / 2, -field_height / 2, field_height / 2)

        if self.w is None:
            return

        for obj in self.w.bodies:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is Box2D.b2.circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=(255, 0, 0)).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=(0, 0, 0), filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=(255, 0, 0) if obj != self.car_2 else (0, 255, 0))
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=(0, 0, 0), linewidth=2)

        self.viewer.render()

    def draw(self):
        self.render()

    def __state__(self):
        c1_angle_error_target, c1_dist_target = calc_angle_error_and_dist(self.car_1, self.ball.position)
        c1_angle_error_other, c1_dist_other = calc_angle_error_and_dist(self.car_1, self.car_2.position)

        c2_angle_error_target, c2_dist_target = calc_angle_error_and_dist(self.car_2, self.ball.position)
        c2_angle_error_other, c2_dist_other = calc_angle_error_and_dist(self.car_2, self.car_1.position)

        c1_state = np.array(
            [self.car_1.position[0] / field_width, self.car_1.position[1] / field_height, c1_angle_error_target,
             c1_angle_error_other, c1_dist_target, c1_dist_other, velocity(self.car_1) / 100, 0, self.car_1.angle/math.pi, velocity(self.ball, return_vector=True)[0]/100])

        c2_state = np.array(
            [self.car_2.position[0] / field_width, self.car_2.position[1] / field_height, c2_angle_error_target,
             c2_angle_error_other, c2_dist_target, c2_dist_other, velocity(self.car_2) / 100, 1, self.car_2.angle/math.pi, velocity(self.ball, return_vector=True)[0]/100])

        return c1_state, c2_state


if __name__ == "__main__":
    env = CarSoccerEnv()
    env.reset()
    while True:
        env.step([1, 1], [0, 0])
        env.render()
