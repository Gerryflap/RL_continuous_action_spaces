# RL Continuous Action Spaces
## An exploration of continuous action spaces in Reinforcement Learning

This repository is a small experiment with continuous action spaces in reinforcement learning. 

Up until now this repo contains:
- 3 algorithms:
  - Simple Policy Optimization, which maximizes action probabilities weighted by their episodic discounted rewards. For exploration it uses an adjustible normal distribution scale parameter.
  - Simple Policy Optimization With Entropy, which maximizes action probabilities weighted by their episodic discounted rewards. For exploration it uses entropy regularization. Both mean and scale are model outputs.
  - Advantage Actor Critic, which uses 2 networks. The Critic learns to predict the state -> value function, while the Actor learns to act such that the advantage (value improvement over predicted value by Critic) and entropy are maximized. (implementation not directly based on paper)

- 2 environments:
  - `CarEnv`, where the goal is to drive the car to the taget using throttle and steering.
  - `MPCarEnv`, which has 2 agents: blue tries to reach the target and red aims to tag the blue car before it reaches the target.
  
  
## An example of the `MPCarEnv` with the Simple Policy Optimization agents:
![alt text](https://github.com/Gerryflap/RL_continuous_action_spaces/blob/master/MPCarEnv.gif?raw=true "MPCarEnv")
