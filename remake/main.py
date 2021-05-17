import gym
import numpy as np
from ppo_torch import Agent
from newmaze import Maze
from utils import plot_learning_curve

if __name__ == '__main__':
    #env = gym.make('CartPole-v0')
    #env = gym.make('FrozenLake-v0', is_slippery=False )
    env = Maze()
    #env = Maze(8)
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    #import pdb; breakpoint()
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(1,)) # frozen lake
    #agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape) # cartpole

    n_games = 1000

    figure_file = 'plots/frozenlake.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


