"""
    Applies the SPO-RNN algorithm to the MPCarEnv.
    Results: X
"""

import time

import numpy as np
import tensorflow as tf
from algorithms import simple_policy_optimization_rnn as spornn
from environments import multiplayer_car_env

ks = tf.keras

SEED = 420
tf.set_random_seed(SEED)
multiplayer_car_env.set_random_seed(SEED)


def make_model():
    inp = ks.Input((None, 7))
    state_inp = ks.Input((32,))

    mem_out, new_rnn_state = ks.layers.GRU(32, return_sequences=True, return_state=True)([inp, state_inp])
    action_means = ks.layers.TimeDistributed(ks.layers.Dense(2, activation='tanh'))(mem_out)
    model = ks.models.Model(inputs=[inp, state_inp], outputs=[action_means, new_rnn_state])
    return model


initial_rnn_state = np.zeros((1, 32))
policy_1 = spornn.SimplePolicyOptimizerRNN(make_model(), 2, initial_rnn_state, scale_value=0.003, gamma=0.9, lr=0.0005)
policy_2 = spornn.SimplePolicyOptimizerRNN(make_model(), 2, initial_rnn_state, scale_value=0.001, gamma=0.9, lr=0.0005)
env = multiplayer_car_env.MPCarEnv(
    allow_red_to_enter_target_zone=False,
    force_fair_game=True,
    speed_limits=(-2, 20),
    throttle_scale=2.0,
    steer_scale=0.5
)

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
            if episode % 50 == 0 and episode != 0:
                time.sleep(1 / 60)
                env.draw()
            score_1 += r1
            score_2 += r2

        policy_1.reset_state()
        policy_2.reset_state()
        policy_1.train(sess, trajectory_1, scale)
        policy_2.train(sess, trajectory_2, scale)

        scale *= 0.999
        if scale < 0.2:
            scale = 0.2
        print(scale, score_1, score_2)
        episode += 1
