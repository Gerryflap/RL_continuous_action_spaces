"""
    The Car -> Target game for AAC.
    Results: With these parameters the environment is almost completely solved before the first rendered run
"""

import time

import tensorflow as tf
from algorithms import advantage_actor_critic as aac
from environments import car_env

ks = tf.keras

SEED = 420
tf.set_random_seed(SEED)
car_env.set_random_seed(SEED)


def create_policy_model():
    inp = ks.Input((6,))
    x = inp
    x = ks.layers.Dense(128, activation='tanh')(x)
    x = ks.layers.Dense(64, activation='tanh')(x)
    means = ks.layers.Dense(2, activation='tanh')(x)
    scales = ks.layers.Dense(2, activation='sigmoid')(x)
    model = ks.Model(inputs=inp, outputs=[means, scales])
    return model


def create_value_model():
    inp = ks.Input((6,))
    x = inp
    x = ks.layers.Dense(128, activation='tanh')(x)
    x = ks.layers.Dense(64, activation='tanh')(x)
    value = ks.layers.Dense(1, activation='linear')(x)
    model = ks.Model(inputs=inp, outputs=value)
    return model


policy = aac.AdvantageActorCritic(create_policy_model(), create_value_model(), 2, entropy_factor=0.1, gamma=0.9, lr=0.0001)
env = car_env.CarEnv(True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    episode = 0
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
            if episode%50 == 0 and episode != 0:
                time.sleep(1/60)
                env.draw()
        entropy = policy.train(sess, trajectory)

        print(score, entropy)
        episode += 1
