"""
    Simple policy optimization algorithm with entropy
    - Learns the scale parameter
    - Maximizes expected discounted reward and policy entropy
"""
import tensorflow as tf
import numpy as np


class SimplePolicyOptimizerWithEntropy(object):
    def __init__(self, model: tf.keras.Model, n_actions, lr=0.001, gamma=0.99, entropy_factor=0.1):
        """
        Initializes the Simple Policy Optimizer With Entropy
        :param model: A Keras model that takes the state as input and outputs action means and action scales as a
                        list of 2 outputs. This can be done using the Keras functional API
        :param n_actions: The number of continuous actions the environment uses
        :param lr: The learning rate used by the Adam optimizer
        :param gamma: Discount factor used in the computation of the expected discounted reward
        :param entropy_factor: Is multiplied in the loss function with the summed entropy.
            Lower values will encourage exploitation, higher values will encourage exploration.
        """
        self.gamma = gamma

        # The input states matrix
        self.states = model.input

        # The mean and scale action values outputted by the model
        assert isinstance(model.output, list) and len(model.output) == 2
        self.action_means = model.output[0]
        self.action_scales = model.output[1]

        # The summed discounted rewards, used to weigh the action probabilities in the loss function
        self.summed_discounted_rewards = tf.placeholder(shape=(None,), dtype=tf.float32)

        # A placeholder to model the actual taken actions. These are used in the loss function
        self.actions_taken = tf.placeholder(shape=(None, n_actions), dtype=tf.float32)

        # The policy distribution
        self.policy_dist = tf.distributions.Normal(self.action_means, self.action_scales)

        # A tensor for sampling actions for given states
        self.sampled_actions = self.policy_dist.sample()

        # The total log probability that a given action was sampled for the given states
        self.action_log_probs = tf.reduce_sum(self.policy_dist.log_prob(self.actions_taken), axis=1)

        # Define the entropy terms
        self.entropy = self.policy_dist.entropy()
        self.mean_entropy = tf.reduce_mean(self.entropy)
        self.summed_entropy = tf.reduce_sum(self.entropy)

        # The energy to be maximized (and the respective loss to be minimized)
        self.score_energy = tf.reduce_sum(self.action_log_probs * self.summed_discounted_rewards)
        self.energy = self.score_energy + entropy_factor * self.summed_entropy

        self.loss = -1 * self.energy

        # The optimizer and optimization step tensor:
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

        # Go backwards over the trajectory and calculate all the discounted rewards using dynamic programming
        for i in range(len(trajectory) - 1, -1, -1):
            discounted_reward = trajectory[i][2] + self.gamma * discounted_reward
            states.append(trajectory[i][0])
            actions.append(trajectory[i][1])
            values.append(discounted_reward)

        # Reverse all lists and make them numpy arrays
        states = np.stack(list(reversed(states)), axis=0)
        actions = np.stack(list(reversed(actions)), axis=0)
        values = np.stack(list(reversed(values)), axis=0)
        return states, actions, values

    def train(self, sess: tf.Session, trajectory):
        """
        Does one training step using the trajectory
        :param scale_value: Alternative scale value for the distribution
        :param sess: The TF session
        :param trajectory: a list of (state, action, reward). It is assumed that the trajectory is one episode!
        :return: The mean entropy for debug purposes
        """

        # Calculate the state values for this specific trajectory
        states, actions, values = self._generate_batch(trajectory)

        # Do one optimization step
        _, mean_entropy_value = sess.run((self.optimize_op, self.mean_entropy), feed_dict={
            self.actions_taken: actions,
            self.states: states,
            self.summed_discounted_rewards: values,
        })
        return mean_entropy_value

    def get_actions(self, sess, state):
        """
        Samples the action values from the policy for the given state
        :param sess: The TF session
        :param state: The current state (without the batch dimension!)
        :return:
        """

        # Add the bach dim
        state = np.expand_dims(state, axis=0)

        # Sample the action values based on our current state
        actions = sess.run((self.sampled_actions,), feed_dict={self.states: state})[0][0]
        return actions
