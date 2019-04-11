"""
    Applies the algorithms to the SoccerEnv.
    Results: The parameters below do result in decent policies, although very different compared to SPO.
        Balance issues may occur
"""
import random
import time
from collections import defaultdict

import matplotlib
import numpy as np
import tensorflow as tf
from algorithms import advantage_actor_critic as aac, random_agent, soccer_agent, linear_agent_switch_wrapper, \
    beta_advantage_actor_critic as baac
from algorithms import simple_policy_optimization as spo
from algorithms import simple_policy_optimization_with_entropy as spowe
from algorithms import simple_policy_optimization_rnn as spornn
from algorithms import dummy_agent
from environments import multiplayer_car_env
import matplotlib.pyplot as plt
import competition_system.matchmaking_systems as ms

from environments.soccer_env import SoccerEnvironment

ks = tf.keras

inp = ks.Input((14,))
x = inp
x = ks.layers.Dense(256, activation='selu')(x)
x = ks.layers.Dense(128, activation='selu')(x)
x2 = ks.layers.Dense(64, activation='selu')(x)
alphas = ks.layers.Dense(2, activation='softplus')(x2)
betas = ks.layers.Dense(2, activation='softplus')(x2)
p_model = ks.Model(inputs=inp, outputs=[alphas, betas])

x2 = ks.layers.Dense(64, activation='selu')(x)
value = ks.layers.Dense(1, activation='linear')(x2)
v_model = ks.Model(inputs=inp, outputs=value)


agent = baac.BetaAdvantageActorCritic(p_model, v_model, 2, lr=0.0001, gamma=0.99,
                                      entropy_factor=0.00001, log=True, value_loss_scale=0.1, lambd=0.97, ppo_eps=0.2)


env = SoccerEnvironment(add_random=False)

try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        episode = 0

        while True:

            agent_1 = agent
            agent_2 = dummy_agent.DummyAgent(2)

            state_1, state_2 = env.reset()
            done = False
            trajectory_1 = []
            trajectory_2 = []
            score_1 = 0
            score_2 = 0
            while not done:
                actions_1 = agent_1.get_actions(sess, state_1)
                actions_2 = agent_2.get_actions(sess, state_2)
                (new_state_1, new_state_2), (r1, r2), done, _ = env.step(actions_1, actions_2)

                trajectory_1.append((state_1, actions_1, r1))
                trajectory_2.append((state_2, actions_2, r2))

                state_1, state_2 = new_state_1, new_state_2
                if episode % 200 == 0 and episode != 0:
                    time.sleep(1 / 60)
                    env.render()
                score_1 += r1
                score_2 += r2
            agent_1.train(sess, trajectory_1)
            agent_2.train(sess, trajectory_2)

            if score_1 > score_2:
                outcome = 1
            elif score_1 < score_2:
                outcome = 2
            else:
                outcome = 0

            episode += 1

except KeyboardInterrupt:
    pass

