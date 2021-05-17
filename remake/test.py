import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
from ppo_torch import ActorNetwork
from newmaze import Maze
if __name__ == '__main__':
    #env = gym.make('FrozenLake-v0', is_slippery=False )
    env = Maze()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(1,))

    agent.load_models()

    observation = env.reset()
    done = False
    env.render()
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        env.render()

