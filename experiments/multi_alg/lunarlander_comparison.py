"""
    Pendulum Gym env for all algorithms
"""


def run_policy(policy_type, episodes):
    import gym
    import tensorflow as tf
    from algorithms import advantage_actor_critic as aac
    from algorithms import simple_policy_optimization as spo
    from algorithms import simple_policy_optimization_with_entropy as spowe
    from algorithms import simple_policy_optimization_rnn as spornn
    from algorithms import beta_advantage_actor_critic as baac
    import numpy as np
    ks = tf.keras
    SEED = 420
    tf.set_random_seed(SEED)

    def create_policy_model_beta():
        inp = ks.Input((8,))
        x = inp
        x = ks.layers.Dense(64, activation='selu')(x)
        x = ks.layers.Dense(32, activation='selu')(x)
        alphas = ks.layers.Dense(2, activation='softplus')(x)
        betas = ks.layers.Dense(2, activation='softplus')(x)
        model = ks.Model(inputs=inp, outputs=[alphas, betas])
        return model

    def create_policy_model_entropy():
        inp = ks.Input((8,))
        x = inp
        x = ks.layers.Dense(64, activation='selu')(x)
        x = ks.layers.Dense(32, activation='selu')(x)
        means = ks.layers.Dense(2, activation='tanh')(x)
        scales = ks.layers.Dense(2, activation='sigmoid')(x)
        model = ks.Model(inputs=inp, outputs=[means, scales])
        return model

    def create_policy_model_no_entropy():
        inp = ks.Input((8,))
        x = inp
        x = ks.layers.Dense(64, activation='selu')(x)
        x = ks.layers.Dense(32, activation='selu')(x)
        means = ks.layers.Dense(2, activation='tanh')(x)
        model = ks.Model(inputs=inp, outputs=means)
        return model

    def create_value_model():
        inp = ks.Input((8,))
        x = inp
        x = ks.layers.Dense(64, activation='selu')(x)
        x = ks.layers.Dense(32, activation='selu')(x)
        value = ks.layers.Dense(1, activation='linear')(x)
        model = ks.Model(inputs=inp, outputs=value)
        return model

    def make_rnn_model():
        inp = ks.Input((None, 8))
        state_inp = ks.Input((32,))

        mem_out, new_rnn_state = ks.layers.GRU(32, return_sequences=True, return_state=True)([inp, state_inp])
        mem_out = ks.layers.TimeDistributed(ks.layers.Dense(32, activation='selu'))(mem_out)
        action_means = ks.layers.TimeDistributed(ks.layers.Dense(2, activation='tanh'))(mem_out)
        model = ks.models.Model(inputs=[inp, state_inp], outputs=[action_means, new_rnn_state])
        return model

    if policy_type == "ac":
        policy = aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 2, entropy_factor=0.001,
                                          gamma=0.99,
                                          lr=0.001)
    elif policy_type == "spowe":
        policy = spowe.SimplePolicyOptimizerWithEntropy(create_policy_model_entropy(), 2, 0.001, gamma=0.99,
                                                        entropy_factor=0.01)
    elif policy_type == "spornn":
        initial_rnn_state = np.zeros((1, 32))
        policy = spornn.SimplePolicyOptimizerRNN(make_rnn_model(), 2, initial_rnn_state, scale_value=0.6, gamma=0.99,
                                                 lr=0.0001)
    elif policy_type == "baac":
        policy = baac.BetaAdvantageActorCritic(create_policy_model_beta(), create_value_model(), 2,
                                               entropy_factor=0.001,
                                               gamma=0.99,
                                               lr=0.001)
    elif policy_type == "ppo":
        policy = baac.BetaAdvantageActorCritic(create_policy_model_beta(), create_value_model(), 2,
                                               entropy_factor=0.001,
                                               gamma=0.99,
                                               ppo_eps=0.2,
                                               lr=0.001)
    else:
        policy = spo.SimplePolicyOptimizer(create_policy_model_no_entropy(), 2, 0.001, gamma=0.99, scale_value=0.6)

    scores = []

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        env = gym.make("LunarLanderContinuous-v2")
        for episode in range(episodes):
            state = env.reset()
            done = False
            trajectory = []
            score = 0
            while not done:
                actions = policy.get_actions(sess, state)
                new_state, r, done, _ = env.step(actions)
                r /= 1000
                score += r

                trajectory.append((state, actions, r))
                state = new_state
            policy.train(sess, trajectory)
            scores.append(score)
            episode += 1
    return scores


import multiprocessing as mp
import matplotlib.pyplot as plt

N_GAMES = 500

policies = [
    "ac",
    "spo",
    "spowe",
    "ppo",
    "baac"
]

policies = zip(policies, [N_GAMES] * len(policies))


def name_and_run(pn):
    p, n = pn
    return p, run_policy(p, n)


if __name__ == "__main__":
    pool = mp.Pool(5)
    results = pool.imap_unordered(name_and_run, policies)
    #results = map(name_and_run, policies)

    for name, scores in results:
        plt.plot(scores, label=name)
    plt.legend()
    plt.show()
