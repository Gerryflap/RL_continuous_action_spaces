"""
    Applies SPO to the CarEnv
    Results: As the scale parameter decays, performance increases. The agent becomes consistent very quickly
"""

import time

import tensorflow as tf
from algorithms import simple_policy_optimization as spo
from environments import car_env

ks = tf.keras

SEED = 420
tf.set_random_seed(SEED)
car_env.set_random_seed(SEED)

model = ks.models.Sequential()
model.add(ks.layers.Dense(24, activation='tanh', input_shape=(6,)))
model.add(ks.layers.Dense(12, activation='tanh'))
model.add(ks.layers.Dense(2, activation='tanh'))

policy = spo.SimplePolicyOptimizer(model, 2, scale_value=0.3, gamma=0.9, lr=0.001)
env = car_env.CarEnv(True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    scale = 3.0
    episode = 0
    while True:
        state = env.reset()
        done = False
        trajectory = []
        while not done:
            actions = policy.get_actions(sess, state, scale)
            new_state, r, done, _ = env.step(actions)

            trajectory.append((state, actions, r))
            state = new_state
            if episode%50 == 0 and episode != 0:
                time.sleep(1/60)
                env.draw()
        policy.train(sess, trajectory, scale)
        scale *= 0.996
        if scale < 0.2:
            scale = 0.2
        print(scale)
        episode += 1
