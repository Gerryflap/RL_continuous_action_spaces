"""
    Advantage Actor Critic (probably not the same as the A2C algorithm described by the paper)
    - Learns the scale parameter
    - Features a policy and a value model
    - Policy maximizes advantage and entropy
    - Value model minimizes the mean squared error with actual returns
"""
import tensorflow as tf
import numpy as np


class BetaAdvantageActorCritic(object):
    def __init__(self, policy_model: tf.keras.Model, value_model: tf.keras.Model, n_actions, lr=0.001, gamma=0.99, entropy_factor=0.1, value_loss_scale=0.5, log=False, lambd=1.0, ppo_eps=None):
        """
        Initializes the Simple Policy Optimizer With Entropy
        :param policy_model: A Keras model that takes the state as input and outputs action means and action scales as a
                        list of 2 outputs. This can be done using the Keras functional API
        :param n_actions: The number of continuous actions the environment uses
        :param lr: The learning rate used by the Adam optimizer
        :param gamma: Discount factor used in the computation of the expected discounted reward
        :param entropy_factor: Is multiplied in the loss function with the summed entropy.
            Lower values will encourage exploitation, higher values will encourage exploration.
        """
        self.gamma = gamma
        self.log = log
        self.lambd = lambd
        self.steps = 0

        # The input states matrix
        self.states_p = policy_model.input
        self.states_v = value_model.input

        # The mean and scale action values outputted by the model
        assert isinstance(policy_model.output, list) and len(policy_model.output) == 2
        self.action_alphas = policy_model.output[0] + 1
        self.action_betas = policy_model.output[1] + 1

        # Predicted values
        self.predicted_values = value_model.output

        # The summed discounted rewards, used to weigh the action probabilities in the loss function
        self.summed_discounted_rewards = tf.placeholder(shape=(None,), dtype=tf.float32)

        # The advantages. If the actual rewards exceed the predicted rewards, this will be positive.
        # Gradients to the value network are stopped to ensure that it is not trained in the policy loss
        self.advantages = self.summed_discounted_rewards - tf.stop_gradient(self.predicted_values)

        # A placeholder to model the actual taken actions. These are used in the loss function
        self.actions_taken = tf.placeholder(shape=(None, n_actions), dtype=tf.float32)
        self.actions_taken_from_dist = (self.actions_taken + 1)/2

        # The policy distribution
        self.policy_dist = tf.distributions.Beta(self.action_alphas, self.action_betas)

        # A tensor for sampling actions for given states
        self.sampled_actions = self.policy_dist.sample() * 2 - 1

        # The total log probability that a given action was sampled for the given states
        self.action_log_probs = tf.reduce_sum(self.policy_dist.log_prob(self.actions_taken_from_dist), axis=1)

        # Define the entropy terms
        self.entropy = self.policy_dist.entropy()
        # self.entropy = tf.log(self.action_scales * (np.pi * np.e * 2)**0.5)# self.policy_dist.entropy()
        self.mean_entropy = tf.reduce_mean(self.entropy)
        self.summed_entropy = tf.reduce_sum(self.entropy)

        self.mean_value = tf.reduce_mean(self.predicted_values)

        tf.summary.scalar("entropy", self.mean_entropy)
        tf.summary.scalar("value", self.mean_value)
        tf.summary.histogram("actions", self.actions_taken)
        tf.summary.scalar("alphas", tf.reduce_mean(self.action_alphas))
        tf.summary.scalar("betas", tf.reduce_mean(self.action_alphas))






        # The energy to be maximized for the policy
        if ppo_eps is None:
            self.p_score_energy = tf.reduce_mean(self.action_log_probs * self.advantages)
        else:
            ratio = self.action_log_probs - tf.stop_gradient(self.action_log_probs)
            ratio = tf.exp(ratio)   # Undo the logprob here in order to do the same as normal ppo
            ratio_with_clip = tf.minimum(ratio, tf.clip_by_value(ratio, 1-ppo_eps, 1+ppo_eps))
            self.p_score_energy = tf.reduce_mean(ratio_with_clip * self.advantages)
        self.p_energy = self.p_score_energy + entropy_factor * self.mean_entropy

        # The loss function for the value network
        self.v_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.summed_discounted_rewards, self.predicted_values))


        # The total loss function
        self.loss = -1 * self.p_energy + value_loss_scale * self.v_loss

        # The optimizer and optimization step tensor:
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.optimize_op = self.optimizer.minimize(self.loss)
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs/ac")

    def _generate_batch(self, trajectory, sess):
        """
        Generates a batch that can be used to train the agent
        :param trajectory: a list of (state, action, reward). It is assumed that the trajectory is one episode!
        :return: states, actions, values (expected discounted reward for that state)
        """
        discounted_reward = 0
        states = []
        actions = []
        values = []

        for i in range(len(trajectory)):
            states.append(trajectory[i][0])
            actions.append(trajectory[i][1])

        states = np.stack(list(states), axis=0)
        actions = np.stack(list(actions), axis=0)

        if self.lambd == 1.0:
            predicted_values = None
        else:
            predicted_values = sess.run(self.predicted_values, feed_dict={self.states_v: states})[:, 0]

        # Go backwards over the trajectory and calculate all the discounted rewards using dynamic programming
        for i in range(len(trajectory) - 1, -1, -1):
            if self.lambd != 1.0:
                pred_v = 0 if i == len(trajectory) - 1 else predicted_values[i+1]
            else:
                pred_v = 0
            discounted_reward_mc = self.gamma * discounted_reward
            discounted_reward_td = self.gamma * pred_v
            discounted_reward = trajectory[i][2] + self.lambd * discounted_reward_mc + (1 - self.lambd) * discounted_reward_td
            values.append(discounted_reward)

        # Reverse the list and make it a numpy array
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
        states, actions, values = self._generate_batch(trajectory, sess)

        # Do one optimization step
        if self.log:
            _, mean_entropy_value, summary = sess.run((self.optimize_op, self.mean_entropy, self.summary_op), feed_dict={
                self.actions_taken: actions,
                self.states_p: states,
                self.states_v: states,
                self.summed_discounted_rewards: values,
            })
            self.writer.add_summary(summary, self.steps)
            self.steps += 1
        else:
            _, mean_entropy_value = sess.run((self.optimize_op, self.mean_entropy), feed_dict={
                self.actions_taken: actions,
                self.states_p: states,
                self.states_v: states,
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
        actions, entropy = sess.run((self.sampled_actions, self.entropy), feed_dict={self.states_p: state})
        actions = actions[0]
        # print(actions, means)
        return actions
