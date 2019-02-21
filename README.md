# RL_continuous_action_spaces
## An exploration of continuous action spaces in Reinforcement Learning

This repository is a small experiment with continuous action spaces in reinforcement learning. 

Up until now this repo contains:
- 1 algorithm:
  - Simple Policy Optimization, which maximizes action probabilities weighted by their episodic discounted rewards. For exploration it uses an adjustible normal distribution scale parameter.

- 2 environments:
  - `CarEnv`, where the goal is to drive the car to the taget using throttle and steering.
  - `MPCarEnv`, which has 2 agents: blue tries to reach the target and red aims to tag the blue car before it reaches the target.
  
  
