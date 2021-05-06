from ppo import PPO

import gym
env = gym.make('Pendulum-v0')
model = PPO(env)
model.learn()