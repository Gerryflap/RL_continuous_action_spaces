import random
import multiprocessing as mp
from threading import Thread

from environments.soccer_env import SoccerEnvironment

PORT = random.randint(10000, 20000)


def run_policy(policy_type):
    from algorithms import dummy_agent
    import tensorflow as tf
    from algorithms import advantage_actor_critic as aac
    from algorithms import simple_policy_optimization as spo
    from algorithms import simple_policy_optimization_with_entropy as spowe
    from algorithms import simple_policy_optimization_rnn as spornn
    import numpy as np
    from algorithms import random_agent
    from competition_system.player import Player
    ks = tf.keras

    # SEED = 420
    # tf.set_random_seed(SEED)
    # car_env.set_random_seed(SEED)

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

    if policy_type == "ac":
        policy = aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 2, entropy_factor=0.01,
                                          gamma=0.997,
                                          lr=0.0001)
    elif policy_type == "spowe":
        policy = spowe.SimplePolicyOptimizerWithEntropy(create_policy_model_entropy(), 2, 0.0001, gamma=0.997,
                                                        entropy_factor=0.01)
    elif policy_type == "spornn":
        initial_rnn_state = np.zeros((1, 32))
        policy = spornn.SimplePolicyOptimizerRNN(make_rnn_model(), 2, initial_rnn_state, scale_value=0.6, gamma=0.997,
                                                 lr=0.0001)
    elif policy_type == "spo":
        policy = spo.SimplePolicyOptimizer(create_policy_model_no_entropy(), 2, 0.0001, gamma=0.997, scale_value=0.6)

    elif policy_type == "random":
        policy = random_agent.RandomAgent(2)

    else:
        policy = dummy_agent.DummyAgent(2)

    scores = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        env = Player("%s" % str(policy), port=PORT)
        while True:
            state = env.reset()
            done = False
            trajectory = []
            score = 0
            while not done:
                actions = policy.get_actions(sess, state)
                new_state, r, done, _ = env.step(actions)

                score += r

                trajectory.append((state, actions, r))
                state = new_state
            policy.train(sess, trajectory)
            scores.append(score)


def run_server():
    import competition_system.normal_env_game as neg
    import competition_system.matchmaking_systems as ms
    from competition_system.competition_server import CompetitionServer

    def env_builder():
        return SoccerEnvironment(gui=False, add_random=False)

    play_game_fn = neg.get_runner(env_builder, log=True)
    server = CompetitionServer(play_game_fn, ms.ScaledMatchMakingSystem(), port=PORT)
    server.run()


server_thread = mp.Process(target=run_server)
server_thread.start()
algos = list([random.choice(["spo", "spowe", "spornn", "ac", "random", "dummy"]) for i in range(12)])

for algo in algos:
    process = mp.Process(target=run_policy, args=(algo,))
    process.start()

server_thread.join()
