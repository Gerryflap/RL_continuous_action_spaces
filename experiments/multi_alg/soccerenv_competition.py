"""
    Applies the AAC algorithm to the MPCarEnv.
    Results: The parameters below do result in decent policies, although very different compared to SPO.
        Balance issues may occur
"""
import random
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from algorithms import advantage_actor_critic as aac, random_agent
from algorithms import simple_policy_optimization as spo
from algorithms import simple_policy_optimization_with_entropy as spowe
from algorithms import simple_policy_optimization_rnn as spornn
from algorithms import dummy_agent
from environments import multiplayer_car_env
import matplotlib.pyplot as plt

from environments.soccer_env import SoccerEnvironment

ks = tf.keras

SEED = 420
random.seed(SEED)
tf.set_random_seed(SEED)
multiplayer_car_env.set_random_seed(SEED)


def create_policy_model_entropy():
    inp = ks.Input((12,))
    x = inp
    x = ks.layers.Dense(64, activation='tanh')(x)
    x = ks.layers.Dense(32, activation='tanh')(x)
    means = ks.layers.Dense(2, activation='tanh')(x)
    scales = ks.layers.Dense(2, activation='sigmoid')(x)
    model = ks.Model(inputs=inp, outputs=[means, scales])
    return model


def create_policy_model_no_entropy():
    inp = ks.Input((12,))
    x = inp
    x = ks.layers.Dense(64, activation='tanh')(x)
    x = ks.layers.Dense(32, activation='tanh')(x)
    means = ks.layers.Dense(2, activation='tanh')(x)
    model = ks.Model(inputs=inp, outputs=means)
    return model


def create_value_model():
    inp = ks.Input((12,))
    x = inp
    x = ks.layers.Dense(64, activation='tanh')(x)
    x = ks.layers.Dense(32, activation='tanh')(x)
    value = ks.layers.Dense(1, activation='linear')(x)
    model = ks.Model(inputs=inp, outputs=value)
    return model


def make_rnn_model():
    inp = ks.Input((None, 12))
    state_inp = ks.Input((32,))

    mem_out, new_rnn_state = ks.layers.GRU(32, return_sequences=True, return_state=True)([inp, state_inp])
    mem_out = ks.layers.TimeDistributed(ks.layers.Dense(32, activation='tanh'))(mem_out)
    action_means = ks.layers.TimeDistributed(ks.layers.Dense(2, activation='tanh'))(mem_out)
    model = ks.models.Model(inputs=[inp, state_inp], outputs=[action_means, new_rnn_state])
    return model


initial_rnn_state = np.zeros((1, 32))
agents = [
    aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 2, lr=0.001, gamma=0.997,
                             entropy_factor=1.0),
    aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 2, lr=0.0001, gamma=0.997,
                             entropy_factor=1.0),
    aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 2, lr=0.00001, gamma=0.997,
                             entropy_factor=1.0),
    spowe.SimplePolicyOptimizerWithEntropy(create_policy_model_entropy(), 2, 0.0001, gamma=0.997,
                                           entropy_factor=1.0),
    spornn.SimplePolicyOptimizerRNN(make_rnn_model(), 2, initial_rnn_state, scale_value=0.6, gamma=0.997,
                                    lr=0.0001),
    spornn.SimplePolicyOptimizerRNN(make_rnn_model(), 2, initial_rnn_state, scale_value=1.2, gamma=0.997,
                                    lr=0.0001),
    spo.SimplePolicyOptimizer(create_policy_model_no_entropy(), 2, 0.0001, gamma=0.997, scale_value=0.6),
    dummy_agent.DummyAgent(2),
    random_agent.RandomAgent(2)
]



def name(agent):
    return str(agent.__class__.__name__) + str(id(agent))


#env = multiplayer_car_env.MPCarEnv(force_fair_game=False, max_steps=500)
env = SoccerEnvironment(add_random=False)

avg_scores = defaultdict(lambda: (0, 0))

scores = defaultdict(lambda: [])


try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        episode = 0
        while True:
            # Pick random agents
            agent_1, agent_2 = random.choice(agents), random.choice(agents)
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

            sum_scores, total_games = avg_scores[name(agent_1)]
            avg_scores[name(agent_1)] = sum_scores + score_1, total_games + 1
            sum_scores, total_games = avg_scores[name(agent_2)]
            avg_scores[name(agent_2)] = sum_scores + score_2, total_games + 1

            scores[name(agent_1)].append(avg_scores[name(agent_1)][0]/avg_scores[name(agent_1)][1])
            scores[name(agent_2)].append(avg_scores[name(agent_2)][0]/avg_scores[name(agent_2)][1])


            for agent in [agent_1, agent_2]:
                if isinstance(agent, spornn.SimplePolicyOptimizerRNN):
                    agent.reset_state()

            # print(score_1, "\t", score_2)

            if episode % 200 == 0:
                print("Displayed game: %s vs. %s"%(name(agent_1), name(agent_2)))

                print("EPISODE ", episode)
                print("AVG SCORES:")
                for agnt, (scr, eps) in avg_scores.items():
                    print("%s: %.3f"%(agnt, scr/eps))

            episode += 1

except KeyboardInterrupt:
    pass

for agnt, scores in scores.items():
    plt.plot(scores, label=agnt)
plt.legend()
plt.title("Blue scores")
plt.show()
