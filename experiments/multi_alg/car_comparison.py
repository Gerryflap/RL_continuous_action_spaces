"""
    The Car -> Target game for all algorithms
"""


def run_policy(policy_type, episodes):
    from environments import car_env
    import tensorflow as tf
    from algorithms import advantage_actor_critic as aac
    from algorithms import simple_policy_optimization as spo
    from algorithms import simple_policy_optimization_with_entropy as spowe
    from algorithms import simple_policy_optimization_rnn as spornn
    import numpy as np
    ks = tf.keras
    SEED = 420
    tf.set_random_seed(SEED)
    car_env.set_random_seed(SEED)

    def create_policy_model_entropy():
        inp = ks.Input((6,))
        x = inp
        x = ks.layers.Dense(128, activation='tanh')(x)
        x = ks.layers.Dense(64, activation='tanh')(x)
        means = ks.layers.Dense(2, activation='tanh')(x)
        scales = ks.layers.Dense(2, activation='sigmoid')(x)
        model = ks.Model(inputs=inp, outputs=[means, scales])
        return model

    def create_policy_model_no_entropy():
        inp = ks.Input((6,))
        x = inp
        x = ks.layers.Dense(128, activation='tanh')(x)
        x = ks.layers.Dense(64, activation='tanh')(x)
        means = ks.layers.Dense(2, activation='tanh')(x)
        model = ks.Model(inputs=inp, outputs=means)
        return model

    def create_value_model():
        inp = ks.Input((6,))
        x = inp
        x = ks.layers.Dense(128, activation='tanh')(x)
        x = ks.layers.Dense(64, activation='tanh')(x)
        value = ks.layers.Dense(1, activation='linear')(x)
        model = ks.Model(inputs=inp, outputs=value)
        return model

    def make_rnn_model():
        inp = ks.Input((None, 6))
        state_inp = ks.Input((128,))

        mem_out, new_rnn_state = ks.layers.GRU(128, return_sequences=True, return_state=True)([inp, state_inp])
        mem_out = ks.layers.TimeDistributed(ks.layers.Dense(64, activation='tanh'))(mem_out)
        action_means = ks.layers.TimeDistributed(ks.layers.Dense(2, activation='tanh'))(mem_out)
        model = ks.models.Model(inputs=[inp, state_inp], outputs=[action_means, new_rnn_state])
        return model

    if policy_type == "ac":
        policy = aac.AdvantageActorCritic(create_policy_model_entropy(), create_value_model(), 2, entropy_factor=0.01,
                                 gamma=0.99,
                                 lr=0.0001)
    elif policy_type == "spowe":
        policy = spowe.SimplePolicyOptimizerWithEntropy(create_policy_model_entropy(), 2, 0.0001, gamma=0.99,
                                               entropy_factor=0.01)
    elif policy_type == "spornn":
        initial_rnn_state = np.zeros((1, 128))
        policy = spornn.SimplePolicyOptimizerRNN(make_rnn_model(), 2, initial_rnn_state, scale_value=0.6, gamma=0.99,
                                                   lr=0.0001)
    else:
        policy = spo.SimplePolicyOptimizer(create_policy_model_no_entropy(), 2, 0.0001, gamma=0.99, scale_value=0.6)


    scores = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        env = car_env.CarEnv(True)
        for episode in range(episodes):
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
            episode += 1
    return scores



import multiprocessing as mp
import matplotlib.pyplot as plt


N_GAMES = 1000

policies = [
    "ac",
    "spo",
    "spowe",
    "spornn"
]

policies = zip(policies, [N_GAMES] * len(policies))

def name_and_run(pn):
    p, n = pn
    return p, run_policy(p, n)


if __name__ == "__main__":
    pool = mp.Pool(4)
    results = pool.imap_unordered(name_and_run, policies)

    for name, scores in results:
        plt.plot(scores, label=name)
    plt.legend()
    plt.show()
