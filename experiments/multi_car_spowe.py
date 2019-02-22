import time

import tensorflow as tf
from algorithms import simple_policy_optimization_with_entropy as spowe
from environments import multiplayer_car_env

ks = tf.keras

SEED = 420
tf.set_random_seed(SEED)
multiplayer_car_env.set_random_seed(SEED)


def create_model():
    inp = ks.Input((7,))
    x = inp
    x = ks.layers.Dense(24, activation='tanh')(x)
    x = ks.layers.Dense(12, activation='tanh')(x)
    means = ks.layers.Dense(2, activation='tanh')(x)
    scales = ks.layers.Dense(2, activation='sigmoid')(x)
    model = ks.Model(inputs=inp, outputs=[means, scales])
    return model


model1 = create_model()
model2 = create_model()

policy_1 = spowe.SimplePolicyOptimizerWithEntropy(model1, 2, entropy_factor=0.03, gamma=0.9, lr=0.001)
policy_2 = spowe.SimplePolicyOptimizerWithEntropy(model2, 2, entropy_factor=0.03, gamma=0.9, lr=0.001)
env = multiplayer_car_env.MPCarEnv()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode = 0
    while True:
        state_1, state_2 = env.reset()
        done = False
        trajectory_1 = []
        trajectory_2 = []
        score_1 = 0
        score_2 = 0
        while not done:
            actions_1 = policy_1.get_actions(sess, state_1)
            actions_2 = policy_2.get_actions(sess, state_2)
            (new_state_1, new_state_2), (r1, r2), done, _ = env.step(actions_1, actions_2)

            trajectory_1.append((state_1, actions_1, r1))
            trajectory_2.append((state_2, actions_2, r2))

            state_1, state_2 = new_state_1, new_state_2
            if episode%50 == 0 and episode != 0:
                time.sleep(1/60)
                env.draw()
            score_1 += r1
            score_2 += r2
        me_1 = policy_1.train(sess, trajectory_1)
        me_2 = policy_2.train(sess, trajectory_2)

        print("Mean entropy: p1=%.4f, p2=%.4f, \tScores: p1=%.2f, p2=%.2f"%(me_1, me_2, score_1, score_2))
        episode += 1
