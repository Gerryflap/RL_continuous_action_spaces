"""
    The Car -> Target game for BAAC.
    Results: With these parameters the environment is almost completely solved before the first rendered run
"""

import time

import tensorflow as tf
from algorithms import beta_advantage_actor_critic as baac
from environments import car_env

ks = tf.keras

SEED = 420
tf.set_random_seed(SEED)
car_env.set_random_seed(SEED)


def create_policy_model_beta():
    inp = ks.Input((6,))
    x = inp
    x = ks.layers.Dense(128, activation='selu')(x)
    x = ks.layers.Dense(64, activation='selu')(x)
    alphas = ks.layers.Dense(2, activation='softplus')(x)
    betas = ks.layers.Dense(2, activation='softplus')(x)
    model = ks.Model(inputs=inp, outputs=[alphas, betas])
    return model


def create_value_model():
    inp = ks.Input((6,))
    x = inp
    x = ks.layers.Dense(128, activation='selu')(x)
    x = ks.layers.Dense(64, activation='selu')(x)
    value = ks.layers.Dense(1, activation='linear')(x)
    model = ks.Model(inputs=inp, outputs=value)
    return model


policy = baac.BetaAdvantageActorCritic(create_policy_model_beta(), create_value_model(), 2, entropy_factor=0.003, gamma=0.97, lr=0.0004, lambd=0.99, value_loss_scale=0.3, log=True)
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
            if episode%500 == 0 and episode != 0:
                time.sleep(1/60)
                env.draw()
        entropy = policy.train(sess, trajectory)

        print(score, entropy)
        episode += 1
