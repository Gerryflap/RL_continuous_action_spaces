"""
    Applies the algorithms to the SoccerEnv.
    Results: The parameters below do result in decent policies, although very different compared to SPO.
        Balance issues may occur
"""
import random
import threading
import time
from collections import defaultdict

import matplotlib
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
from algorithms import advantage_actor_critic as aac, random_agent, soccer_agent, linear_agent_switch_wrapper, \
    beta_advantage_actor_critic as baac, carsoccer_agent
from algorithms import simple_policy_optimization as spo
from algorithms import simple_policy_optimization_with_entropy as spowe
from algorithms import simple_policy_optimization_rnn as spornn
from algorithms import dummy_agent
from environments import multiplayer_car_soccer_env, box2d_car_soccer_env
import matplotlib.pyplot as plt
import competition_system.matchmaking_systems as ms

from environments.soccer_env import SoccerEnvironment

ks = tf.keras

# This option is currently broken and should stay on false
terminate_without_terminal_state = False

action_repeat_frames = 4
max_steps = 50
batch_size = 10
# In number of episodes: (WARNING: Tensorboard shows number of training updates, which is episodes/batch_size)
agent_cloning_interval = 5000
# Number of episodes before agent switching will begin (useful for pre-training)
warmup_episodes = 20000
agent_picking_chance = 0.3
max_agents = 20
gamma = 0.97
warmup_backwards_cost = 0.01
# Add something here to make the filename unique if multiple experiments are being run:
extra_name_addition = "rand_warmup"


# Code starts here
fps_lock = False
show_screen = False

def command_loop():
    global fps_lock
    global show_screen
    while True:
        inp = input("Enter command (help for help): ")
        if inp == "help":
            print("Commands: \n\thelp: displays help.\n\tslow: locks fps to 60 when drawing\n\tfast: unlocks fps when drawing")
        elif inp == "slow":
            fps_lock = True
            print("locking fps...")
        elif inp == "fast":
            fps_lock = False
            print("unlocking fps...")
        elif inp == "show":
            show_screen = True
            print("drawing started.")
        elif inp == "stop show":
            print("drawing is stopped.")
            show_screen = False


threading.Thread(target=command_loop).start()

def make_models():
    inp = ks.Input((11,))
    x = inp
    x = ks.layers.Dense(128)(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Activation('selu')(x)

    x = ks.layers.Dense(64)(x)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Activation('selu')(x)

    x2 = ks.layers.Dense(32)(x)
    x2 = ks.layers.BatchNormalization()(x2)
    x2 = ks.layers.Activation('selu')(x2)
    alphas = ks.layers.Dense(2, activation='softplus')(x2)
    betas = ks.layers.Dense(2, activation='softplus')(x2)
    p_model = ks.Model(inputs=inp, outputs=[alphas, betas])

    x2 = ks.layers.Dense(32)(x)
    x2 = ks.layers.BatchNormalization()(x2)
    x2 = ks.layers.Activation('selu')(x2)
    value = ks.layers.Dense(1, activation='linear')(x2)
    v_model = ks.Model(inputs=inp, outputs=value)
    return p_model, v_model

p_model, v_model = make_models()



log_name = "ac_cs_%f"%gamma
if terminate_without_terminal_state:
    log_name += "_no_term"

if action_repeat_frames > 1:
    log_name += "_rep_%d"%(action_repeat_frames,)

if extra_name_addition:
    log_name += "_" + extra_name_addition

agent = baac.BetaAdvantageActorCritic(p_model, v_model, 2, entropy_factor=0.0005, gamma=gamma, lr=0.0004, lambd=0.99, value_loss_scale=0.001, ppo_eps=0.2, log=True, log_name=log_name)

def clone_agent(agent):
    p_model, v_model = make_models()
    p_model.set_weights(agent.p_model.get_weights())
    v_model.set_weights(agent.v_model.get_weights())
    new_agent = baac.BetaAdvantageActorCritic(p_model, v_model, 2, log=False)
    return new_agent




# Previously max_steps = 10
env = box2d_car_soccer_env.CarSoccerEnv(max_steps=-1 if terminate_without_terminal_state else max_steps*action_repeat_frames, distance_to_ball_r_factor=0.0, distance_to_goal_r_factor=0.0)


# def modify_actions(actions):
#     if actions[0] > -0.1:
#         actions[0] = (actions[0] + 0.1)/1.1
#     else:
#         actions[0] = (actions[0] + 0.1)/0.9
#     return actions

old_agents = []
previous_agent = clone_agent(agent)


def pick_random_agent(episode):
    if not old_agents:
        # If there are no older agents, just return the clone made at the start
        return previous_agent
    else:
        # Compute the probability of picking the previous agent. This should increase to the agent_picking_chance slowly over the agent_cloning_interval
        p = agent_picking_chance * ((episode%agent_cloning_interval)/agent_cloning_interval)
        if random.random() < p:
            return previous_agent
        # Enter picking loop (the list is traversed in the other direction in order to let the most recent have the highest chance
        for agent in old_agents[::-1]:
            if agent_picking_chance > random.random():
                return agent

        # If the list is empty and none is picked, return the last one
        return old_agents[-1]


try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        episode = 0

        agents = [agent, clone_agent(agent)]
        # agents = [agent, carsoccer_agent.CarSoccerAgent(2)]
        trajectory = []

        while True:

            # Place a random previous agent in here
            agents[1] = pick_random_agent(episode)
            first = random.choice([0, 1])
            agent_1 = agents[first]
            agent_2 = agents[1-first]

            if warmup_episodes > episode:
                # Use random spawns in pre-training/warmup
                state_1, state_2 = env.reset_random()
            else:
                state_1, state_2 = env.reset()

            done = False

            score_1 = 0
            score_2 = 0

            trajectory_1 = []
            trajectory_2 = []

            step = 0
            while not done:
                actions_1 = agent_1.get_actions(sess, state_1)
                actions_2 = agent_2.get_actions(sess, state_2)

                if np.isnan(actions_1).any() or np.isnan(actions_2).any():
                    print("NaN detected: reverting!")
                    agents[0] = clone_agent(agents[1])
                r1t, r2t = 0, 0
                for _ in range(action_repeat_frames):
                    (new_state_1, new_state_2), (r1, r2), done, _ = env.step(actions_1, actions_2)
                    r1t += r1
                    r2t += r2
                    if show_screen:
                        if fps_lock:
                            time.sleep(1 / 60)
                        env.draw()
                    if done:
                        break
                r1, r2 = r1t, r2t

                trajectory_1.append((state_1, actions_1, r1, done))
                trajectory_2.append((state_2, actions_2, r2, done))

                state_1, state_2 = new_state_1, new_state_2

                score_1 += r1
                score_2 += r2
                step += 1
                if step > max_steps:
                    break

            if first == 0:
                trajectory += trajectory_1
            else:
                trajectory += trajectory_2

            # Change backwards cost after pre-training/warmup
            if episode == warmup_episodes:
                env.reversing_cost = 0.0

            if episode%batch_size == 0:
                if first == 0:
                    agent_1.train(sess, trajectory)
                else:
                    agent_2.train(sess, trajectory)
                trajectory = []


            if score_1 > score_2:
                outcome = 1
            elif score_1 < score_2:
                outcome = 2
            else:
                outcome = 0

            if episode % agent_cloning_interval == 0 and episode >= warmup_episodes:
                old_agents.append(previous_agent)
                if len(old_agents) > max_agents:
                    old_agents = old_agents[-max_agents:]
                previous_agent = clone_agent(agents[0])
                print("Old agents length: ", len(old_agents))
                pass

            # if episode % 20000 == 0:
            #     if max_steps < 2000:
            #         max_steps *= 2
            #
            #         if not terminate_without_terminal_state:
            #             env.max_steps = max_steps * action_repeat_frames

            episode += 1

except KeyboardInterrupt:
    pass


