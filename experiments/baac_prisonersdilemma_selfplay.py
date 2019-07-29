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
from algorithms import beta_advantage_actor_critic as baac
from environments import prisoners_dilemma_env

ks = tf.keras

def make_models():
    inp = ks.Input((2,))
    x = inp
    x = ks.layers.Dense(256, activation='selu')(x)
    x2 = ks.layers.Dense(128, activation='selu')(x)
    alphas = ks.layers.Dense(1, activation='softplus')(x2)
    betas = ks.layers.Dense(1, activation='softplus')(x2)
    p_model = ks.Model(inputs=inp, outputs=[alphas, betas])

    x2 = ks.layers.Dense(128, activation='selu')(x)
    value = ks.layers.Dense(1, activation='linear')(x2)
    v_model = ks.Model(inputs=inp, outputs=value)
    return p_model, v_model

p_model, v_model = make_models()

agent = baac.BetaAdvantageActorCritic(p_model, v_model, 1, entropy_factor=0.01, gamma=0.97, lr=0.0004, lambd=0.99, value_loss_scale=0.01, ppo_eps=0.2, log=True, log_name="ac_prisoners")

def clone_agent(agent):
    p_model, v_model = make_models()
    p_model.set_weights(agent.p_model.get_weights())
    v_model.set_weights(agent.v_model.get_weights())
    new_agent = baac.BetaAdvantageActorCritic(p_model, v_model, 1, log=False)
    return new_agent

# Previously max_steps = 10
env = prisoners_dilemma_env.PrisonersDilemmaEnv()


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

            state_1, state_2 = env.reset()
            done = False
            trajectory_1 = []
            trajectory_2 = []
            score_1 = 0
            score_2 = 0
            while not done:
                actions_1 = agent_1.get_actions(sess, state_1)
                actions_2 = agent_2.get_actions(sess, state_2)

                if np.isnan(actions_1).any() or np.isnan(actions_1).any():
                    print("NaN detected: reverting!")
                    agents[0] = clone_agent(agents[1])
                (new_state_1, new_state_2), (r1, r2), done, _ = env.step(actions_1, actions_2)

                trajectory_1.append((state_1, actions_1, r1))
                trajectory_2.append((state_2, actions_2, r2))

                state_1, state_2 = new_state_1, new_state_2
                if episode % 200 == 0 and episode != 0:
                    # time.sleep(1 / 60)
                    env.draw()
                score_1 += r1
                score_2 += r2
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

            if episode % 100 == 0:
                agents[1] = clone_agent(agents[0])

            episode += 1

except KeyboardInterrupt:
    pass

