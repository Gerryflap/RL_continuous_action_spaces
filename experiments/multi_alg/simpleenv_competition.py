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
from algorithms import advantage_actor_critic as aac, random_agent
from algorithms import simple_policy_optimization as spo
from algorithms import simple_policy_optimization_with_entropy as spowe
from algorithms import simple_policy_optimization_rnn as spornn
from algorithms import dummy_agent
import matplotlib.pyplot as plt
import competition_system.matchmaking_systems as ms
from environments.simple_comp_env import SimpleCompEnv


ks = tf.keras

# SEED = 420
# random.seed(SEED)
# tf.set_random_seed(SEED)
# multiplayer_car_env.set_random_seed(SEED)


def create_policy_model_entropy():
    inp = ks.Input((2,))
    x = inp
    x = ks.layers.Dense(64, activation='selu')(x)
    x = ks.layers.Dense(32, activation='selu')(x)
    means = ks.layers.Dense(1, activation='tanh')(x)
    scales = ks.layers.Dense(1, activation='sigmoid')(x)
    model = ks.Model(inputs=inp, outputs=[means, scales])
    return model


def create_policy_model_no_entropy():
    inp = ks.Input((2,))
    x = inp
    x = ks.layers.Dense(16, activation='selu')(x)
    x = ks.layers.Dense(16, activation='selu')(x)
    x = ks.layers.Dense(8, activation='selu')(x)
    x = ks.layers.Dense(8, activation='selu')(x)
    means = ks.layers.Dense(1, activation='tanh')(x)
    model = ks.Model(inputs=inp, outputs=means)
    return model


def create_value_model():
    inp = ks.Input((2,))
    x = inp
    x = ks.layers.Dense(16, activation='selu')(x)
    x = ks.layers.Dense(16, activation='selu')(x)
    x = ks.layers.Dense(8, activation='selu')(x)
    x = ks.layers.Dense(8, activation='selu')(x)
    value = ks.layers.Dense(1, activation='linear')(x)
    model = ks.Model(inputs=inp, outputs=value)
    return model


def make_rnn_model():
    inp = ks.Input((None, 2))
    state_inp = ks.Input((32,))

    mem_out, new_rnn_state = ks.layers.GRU(32, return_sequences=True, return_state=True)([inp, state_inp])
    mem_out = ks.layers.TimeDistributed(ks.layers.Dense(32, activation='selu'))(mem_out)
    mem_out = ks.layers.TimeDistributed(ks.layers.Dense(16, activation='selu'))(mem_out)
    mem_out = ks.layers.TimeDistributed(ks.layers.Dense(8, activation='selu'))(mem_out)
    action_means = ks.layers.TimeDistributed(ks.layers.Dense(1, activation='tanh'))(mem_out)
    model = ks.models.Model(inputs=[inp, state_inp], outputs=[action_means, new_rnn_state])
    return model


initial_rnn_state = np.zeros((1, 32))
agents = [
    aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 1, lr=0.001, gamma=0.997,
                             entropy_factor=0.1, value_loss_scale=0.03, log=True, lambd=0.95, scale_multiplier=1.0),
    aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 1, lr=0.001, gamma=0.997,
                             entropy_factor=0.1, value_loss_scale=0.03, lambd=0.95, scale_multiplier=1.0),
    # aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 1, lr=0.0001, gamma=0.997,
    #                          entropy_factor=1.0),
    # aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 1, lr=0.0001, gamma=0.997,
    #                          entropy_factor=1.0),
    # aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 1, lr=0.0001, gamma=0.997,
    #                          entropy_factor=0.1),
    # spornn.SimplePolicyOptimizerRNN(make_rnn_model(), 1, initial_rnn_state, scale_value=0.6, gamma=0.997,
    #                                 lr=0.0001),
    # spornn.SimplePolicyOptimizerRNN(make_rnn_model(), 1, initial_rnn_state, scale_value=0.6, gamma=0.997,
    #                                 lr=0.0001),
    # spo.SimplePolicyOptimizer(create_policy_model_no_entropy(), 1, 0.0001, gamma=0.997, scale_value=0.6),
    # spo.SimplePolicyOptimizer(create_policy_model_no_entropy(), 1, 0.0001, gamma=0.997, scale_value=0.6),
    #dummy_agent.DummyAgent(2),
    # random_agent.RandomAgent(2),
]

pids = {agents[i]: i for i in range(len(agents))}

pids_in_queue = set(pids.values())

def name(agent):
    return str(agent.__class__.__name__) + str(id(agent))


#env = multiplayer_car_env.MPCarEnv(force_fair_game=False, max_steps=500)
env = SimpleCompEnv()

avg_scores = defaultdict(lambda: (0, 0))

scores = defaultdict(lambda: [])
episodes = defaultdict(lambda: [])

matchmaking = ms.ScaledMatchMakingSystem()


try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        episode = 0

        match_queue = matchmaking.get_matches(pids_in_queue)
        while True:
            # Pick random agents
            if len(match_queue) == 0:
                match_queue = matchmaking.get_matches(pids_in_queue)
            agent_1_pid, agent_2_pid = match_queue.pop()
            agent_1 = agents[agent_1_pid]
            agent_2 = agents[agent_2_pid]
            # print(str(agent_1.__class__.__name__), "vs. ", str(agent_2.__class__.__name__))

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

            matchmaking.report_outcome(agent_1_pid, agent_2_pid, outcome)

            sum_scores, total_games = avg_scores[name(agent_1)]
            avg_scores[name(agent_1)] = sum_scores + score_1, total_games + 1
            sum_scores, total_games = avg_scores[name(agent_2)]
            avg_scores[name(agent_2)] = sum_scores + score_2, total_games + 1

            scores[name(agent_1)].append(matchmaking.get_rating(agent_1_pid))
            scores[name(agent_2)].append(matchmaking.get_rating(agent_2_pid))
            episodes[name(agent_1)].append(episode)
            episodes[name(agent_2)].append(episode)


            for agent in [agent_1, agent_2]:
                if isinstance(agent, spornn.SimplePolicyOptimizerRNN):
                    agent.reset_state()

            # print(score_1, "\t", score_2)

            if episode % 200 == 0:
                print("Displayed game: %s vs. %s"%(name(agent_1), name(agent_2)))

                print("EPISODE ", episode)
                print("AVG SCORES:")
                for agent, pid in pids.items():
                    agnt = name(agent)
                    scr, eps = avg_scores[agnt]
                    print("%s: %.3f\t\t%.3f"%(agnt, 0 if eps == 0 else scr/eps, matchmaking.get_rating(pid)))

            episode += 1

except KeyboardInterrupt:
    pass

plt.figure(dpi=200)
for agnt, scores in scores.items():
    plt.plot(episodes[agnt], scores, label=agnt)
font = {'family' : 'normal',
        'size'   : 5}

matplotlib.rc('font', **font)
plt.legend()
plt.title("Blue scores")
plt.show()
