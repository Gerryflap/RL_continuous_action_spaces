import gym

"""
    The Car -> Target game for BAAC.
    Results: With these parameters the environment is almost completely solved before the first rendered run
"""

import time

import tensorflow as tf
from algorithms import beta_advantage_actor_critic as baac

ks = tf.keras

SEED = 420
tf.set_random_seed(SEED)


def create_policy_model_beta():
    inp = ks.Input((4,))
    x = inp
    x = ks.layers.Dense(32, activation='selu')(x)
    x = ks.layers.Dense(16, activation='selu')(x)
    alphas = ks.layers.Dense(1, activation='softplus')(x)
    betas = ks.layers.Dense(1, activation='softplus')(x)
    model = ks.Model(inputs=inp, outputs=[alphas, betas])
    return model


def create_value_model():
    inp = ks.Input((4,))
    x = inp
    x = ks.layers.Dense(32, activation='selu')(x)
    x = ks.layers.Dense(16, activation='selu')(x)
    value = ks.layers.Dense(1, activation='linear')(x)
    model = ks.Model(inputs=inp, outputs=value)
    return model

policy = baac.BetaAdvantageActorCritic(create_policy_model_beta(), create_value_model(), 1, entropy_factor=0.003, gamma=0.97, lr=0.001, lambd=0.99, value_loss_scale=0.1)
env = gym.make("CartPole-v1")

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
            new_state, r, done, _ = env.step(actions[0] > 0)

            score += r

            trajectory.append((state, actions, r))
            state = new_state
            if episode%50 == 0 and episode != 0:
                time.sleep(1/60)
                env.render()
        entropy = policy.train(sess, trajectory)

        print(score, entropy)
        episode += 1
