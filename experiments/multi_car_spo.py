import time

import tensorflow as tf
from algorithms import simple_policy_optimization as spo
from environments import multiplayer_car_env

ks = tf.keras

SEED = 420
tf.set_random_seed(SEED)
multiplayer_car_env.set_random_seed(SEED)

model1 = ks.models.Sequential()
model1.add(ks.layers.Dense(24, activation='tanh', input_shape=(7,)))
model1.add(ks.layers.Dense(12, activation='tanh'))
model1.add(ks.layers.Dense(2, activation='tanh'))

model2 = ks.models.Sequential()
model2.add(ks.layers.Dense(24, activation='tanh', input_shape=(7,)))
model2.add(ks.layers.Dense(12, activation='tanh'))
model2.add(ks.layers.Dense(2, activation='tanh'))

policy_1 = spo.SimplePolicyOptimizer(model1, 2, scale_value=0.003, gamma=0.9, lr=0.001)
policy_2 = spo.SimplePolicyOptimizer(model2, 2, scale_value=0.001, gamma=0.9, lr=0.001)
env = multiplayer_car_env.MPCarEnv()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    scale = 1.5
    episode = 0
    while True:
        state_1, state_2 = env.reset()
        done = False
        trajectory_1 = []
        trajectory_2 = []
        score_1 = 0
        score_2 = 0
        while not done:
            actions_1 = policy_1.get_actions(sess, state_1, scale)
            actions_2 = policy_2.get_actions(sess, state_2, scale)
            (new_state_1, new_state_2), (r1, r2), done, _ = env.step(actions_1, actions_2)

            trajectory_1.append((state_1, actions_1, r1))
            trajectory_2.append((state_2, actions_2, r2))

            state_1, state_2 = new_state_1, new_state_2
            if episode%50 == 0 and episode != 0:
                time.sleep(1/60)
                env.draw()
            score_1 += r1
            score_2 += r2
        policy_1.train(sess, trajectory_1, scale)
        policy_2.train(sess, trajectory_2, scale)

        scale *= 0.999
        if scale < 0.2:
            scale = 0.2
        print(scale, score_1, score_2)
        episode += 1
