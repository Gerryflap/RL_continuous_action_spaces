"""
    Simple policy optimization algorithm RNN
    - Uses fixed variance
    - Maximizes expected discounted reward
    - Allows for a policy with memory
"""
import tensorflow as tf
import numpy as np


class SimplePolicyOptimizerRNN(object):
    def __init__(self, model: tf.keras.Model, n_actions, initial_rnn_state, lr=0.001, gamma=0.99, scale_value=0.1):
        """
        Initializes the Simple Policy Optimizer
        :param model: A Keras model that takes [game_state, rnn_state] as input and outputs [action means, new_rnn_state]
        :param n_actions: The number of continuous actions the environment uses
        :param initial_rnn_state: The initial state of the RNN. This should usually be np.zeros((1, rnn_state_size))
        :param lr: The learning rate used by the Adam optimizer
        :param gamma: Discount factor used in the computation of the expected discounted reward
        :param scale_value: The default value for the scale of the normal distributions
        """
        self.gamma = gamma
        self.scale_value = scale_value
        self.initial_rnn_state = initial_rnn_state

        self.rnn_state_value = initial_rnn_state

        # The input states matrix
        self.states = model.input[0]
        self.rnn_states = model.input[1]

        # The mean action values outputted by the model
        self.action_means = model.output[0]
        self.new_rnn_state = model.output[1]

        # The summed discounted rewards, used to weigh the action probabilities in the loss function
        self.summed_discounted_rewards = tf.placeholder(shape=(1, None), dtype=tf.float32)

        # A placeholder to model the actual taken actions. These are used in the loss function
        self.actions_taken = tf.placeholder(shape=(1, None, n_actions), dtype=tf.float32)

        # The scale parameter of the normal distribution. This (scalar) value is given by the user.
        self.scale = tf.placeholder_with_default(scale_value, shape=None)

        # The policy distribution
        self.policy_dist = tf.distributions.Normal(self.action_means, self.scale)

        # A tensor for sampling actions for given states
        self.sampled_actions = self.policy_dist.sample()

        # The total log probability that a given action was sampled for the given states
        self.action_log_probs = tf.reduce_sum(self.policy_dist.log_prob(self.actions_taken), axis=2)

        # The energy to be maximized (and the respective loss to be minimized)
        self.energy = self.action_log_probs * self.summed_discounted_rewards
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
        for i in range(len(trajectory)-1, -1, -1):
            discounted_reward = trajectory[i][2] + self.gamma * discounted_reward
            states.append(trajectory[i][0])
            actions.append(trajectory[i][1])
            values.append(discounted_reward)

        # Reverse all lists and make them numpy arrays
        states = np.expand_dims(np.stack(list(reversed(states)), axis=0), axis=0)
        actions = np.expand_dims(np.stack(list(reversed(actions)), axis=0), axis=0)
        values = np.expand_dims(np.stack(list(reversed(values)), axis=0), axis=0)
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

        # Calculate the state values for this specific trajectory
        states, actions, values = self._generate_batch(trajectory)

        # Do one optimization step
        sess.run((self.optimize_op,), feed_dict={
            self.actions_taken: actions,
            self.states: states,
            self.summed_discounted_rewards: values,
            self.scale: scale_value,
            self.rnn_states: self.initial_rnn_state
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

        # Add the bach dim
        state = np.expand_dims(state, axis=0)

        # Add time dim
        state = np.expand_dims(state, axis=1)

        # Sample the action values based on our current state
        actions, rnn_state = sess.run((self.sampled_actions, self.new_rnn_state),
                                      feed_dict={
                                          self.states: state,
                                          self.scale: scale_value,
                                          self.rnn_states: self.rnn_state_value
                                      }
                                      )
        actions = actions[0, 0]
        self.rnn_state_value = rnn_state
        return actions

    def reset_state(self):
        self.rnn_state_value = self.initial_rnn_state
