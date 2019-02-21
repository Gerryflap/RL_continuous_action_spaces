"""
    Simple policy optimization algorithm
    - Uses fixed variance
    - Maximizes expected discounted reward
"""
import tensorflow as tf
import numpy as np


class SimplePolicyOptimizer(object):
    def __init__(self, model: tf.keras.Model, n_actions, lr=0.001, gamma=0.99, scale_value=0.1):
        """
        Initializes the Simple Policy Optimizer
        :param model: A Keras model that takes the state as input and outputs action means
        :param n_actions: The number of continuous actions the environment uses
        :param lr: The learning rate used by the Adam optimizer
        :param gamma: Discount factor used in the computation of the expected discounted reward
        :param scale_value: The default value for the scale of the normal distributions
        """
        self.gamma = gamma
        self.scale_value=scale_value

        self.states = model.input
        self.action_means = model.output
        self.mean_discounted_rewards = tf.placeholder(shape=(None, ), dtype=tf.float32)
        self.actions_taken = tf.placeholder(shape=(None, n_actions), dtype=tf.float32)
        self.scale = tf.placeholder_with_default(scale_value, shape=None)
        self.policy_dist = tf.distributions.Normal(self.action_means, self.scale)
        self.sampled_actions = self.policy_dist.sample()
        self.action_log_probs = tf.reduce_sum(self.policy_dist.log_prob(self.actions_taken), axis=1)

        self.energy = self.action_log_probs * self.mean_discounted_rewards
        self.loss = -1 * self.energy
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.optimize_op = self.optimizer.minimize(self.loss)

    def _generate_batch(self, trajectory):
        """
        Generates a batch that can be used to train the agent
        :param trajectory: a list of (state, action, reward). It is assumed that the trajectory is one episode!
        :return: states, actions, values (expected discounted reward for that state)
        """
        discounted_reward = 0
        states = []
        actions = []
        values = []
        for i in range(len(trajectory)-1, -1, -1):
            discounted_reward = trajectory[i][2] + self.gamma * discounted_reward
            states.append(trajectory[i][0])
            actions.append(trajectory[i][1])
            values.append(discounted_reward)

        states = np.stack(list(reversed(states)), axis=0)
        actions = np.stack(list(reversed(actions)), axis=0)
        values = np.stack(list(reversed(values)), axis=0)
        return states, actions, values

    def train(self, sess: tf.Session, trajectory, scale_value=None):
        """
        Does one training step using the trajectory
        :param scale_value: Alternative scale value for the distribution
        :param sess: The TF session
        :param trajectory: a list of (state, action, reward). It is assumed that the trajectory is one episode!
        :param scale_value: The scale of the normal distributions (overrides the default if not None)
        :return: Nothing
        """
        if scale_value is None:
            scale_value = self.scale_value

        states, actions, values = self._generate_batch(trajectory)
        sess.run((self.optimize_op,), feed_dict={
            self.actions_taken: actions,
            self.states: states,
            self.mean_discounted_rewards: values,
            self.scale: scale_value
        })

    def get_actions(self, sess, state, scale_value=None):
        """
        Samples the action values from the policy for the given state
        :param sess: The TF session
        :param state: The current state (without the batch dimension!)
        :param scale_value: The scale of the normal distributions (overrides the default if not None)
        :return:
        """
        if scale_value is None:
            scale_value = self.scale_value
        state = np.expand_dims(state, axis=0)
        actions = sess.run((self.sampled_actions, ), feed_dict={self.states: state, self.scale: scale_value})[0][0]
        return actions
