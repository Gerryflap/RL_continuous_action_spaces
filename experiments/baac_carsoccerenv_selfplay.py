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
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
from algorithms import advantage_actor_critic as aac, random_agent, soccer_agent, linear_agent_switch_wrapper, \
    beta_advantage_actor_critic as baac
from algorithms import simple_policy_optimization as spo
from algorithms import simple_policy_optimization_with_entropy as spowe
from algorithms import simple_policy_optimization_rnn as spornn
from algorithms import dummy_agent
from environments import multiplayer_car_soccer_env, box2d_car_soccer_env
import matplotlib.pyplot as plt
import competition_system.matchmaking_systems as ms

from environments.soccer_env import SoccerEnvironment

ks = tf.keras


terminate_without_terminal_state = True
action_repeat_frames = 10

def make_models():
    inp = ks.Input((11,))
    x = inp
    x = ks.layers.Dense(256)(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Activation('selu')(x)

    x = ks.layers.Dense(256)(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Activation('selu')(x)

    x2 = ks.layers.Dense(128)(x)
    x2 = ks.layers.BatchNormalization()(x2)
    x2 = ks.layers.Activation('selu')(x2)
    alphas = ks.layers.Dense(2, activation='softplus')(x2)
    betas = ks.layers.Dense(2, activation='softplus')(x2)
    p_model = ks.Model(inputs=inp, outputs=[alphas, betas])

    x2 = ks.layers.Dense(128)(x)
    x2 = ks.layers.BatchNormalization()(x2)
    x2 = ks.layers.Activation('selu')(x2)
    value = ks.layers.Dense(1, activation='linear')(x2)
    v_model = ks.Model(inputs=inp, outputs=value)
    return p_model, v_model

p_model, v_model = make_models()


log_name = "ac_cs"
if terminate_without_terminal_state:
    log_name += "_no_term"

if action_repeat_frames > 1:
    log_name += "_rep_%d"%(action_repeat_frames,)

agent = baac.BetaAdvantageActorCritic(p_model, v_model, 2, entropy_factor=0.001, gamma=0.997, lr=0.0004, lambd=0.99, value_loss_scale=0.01, ppo_eps=0.2, log=True, log_name=log_name)

def clone_agent(agent):
    p_model, v_model = make_models()
    p_model.set_weights(agent.p_model.get_weights())
    v_model.set_weights(agent.v_model.get_weights())
    new_agent = baac.BetaAdvantageActorCritic(p_model, v_model, 2, log=False)
    return new_agent


max_steps = 60

# Previously max_steps = 10
env = box2d_car_soccer_env.CarSoccerEnv(max_steps=-1 if terminate_without_terminal_state else max_steps, distance_to_ball_r_factor=0.2, distance_to_goal_r_factor=1.0)


# def modify_actions(actions):
#     if actions[0] > -0.1:
#         actions[0] = (actions[0] + 0.1)/1.1
#     else:
#         actions[0] = (actions[0] + 0.1)/0.9
#     return actions

try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        episode = 0

        agents = [agent, clone_agent(agent)]

        while True:
            first = random.choice([0, 1])
            agent_1 = agents[first]
            agent_2 = agents[1-first]

            # state_1, state_2 = env.reset_random()
            state_1, state_2 = env.reset()

            done = False
            trajectory_1 = []
            trajectory_2 = []
            score_1 = 0
            score_2 = 0

            step = 0
            while not done:
                actions_1 = agent_1.get_actions(sess, state_1)
                actions_2 = agent_2.get_actions(sess, state_2)

                if np.isnan(actions_1).any() or np.isnan(actions_1).any():
                    print("NaN detected: reverting!")
                    agents[0] = clone_agent(agents[1])
                for _ in range(action_repeat_frames):
                    (new_state_1, new_state_2), (r1, r2), done, _ = env.step(actions_1, actions_2)
                    if episode % 200 == 0 and episode != 0:
                        # time.sleep(1 / 60)
                        env.draw()
                    if done:
                        break

                trajectory_1.append((state_1, actions_1, r1, done))
                trajectory_2.append((state_2, actions_2, r2, done))

                state_1, state_2 = new_state_1, new_state_2

                score_1 += r1
                score_2 += r2
                step += 1
                if step > max_steps:
                    break
            if first == 0:
                agent_1.train(sess, trajectory_1)
            else:
                agent_2.train(sess, trajectory_2)

            if score_1 > score_2:
                outcome = 1
            elif score_1 < score_2:
                outcome = 2
            else:
                outcome = 0

            if episode % 1000 == 0:
                agents[1] = clone_agent(agents[0])

            if episode % 20000 == 0:
                if max_steps < 2000:
                    max_steps *= 2

                    if not terminate_without_terminal_state:
                        env.max_steps = max_steps * action_repeat_frames

            episode += 1

except KeyboardInterrupt:
    pass

