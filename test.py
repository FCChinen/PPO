import gym
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F

device = "cpu"

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device) / 64.

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim=64, out_dim=4):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        #self.layer3 = nn.Linear(128, out_dim)
        self.pi_logits = nn.Linear(128, out_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, obs):

        h = F.relu(self.layer1(obs))
        h = F.relu(self.layer2(h))
        #output = self.layer3(h)
        
        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h)

        return pi, value

neural = FeedForwardNN(1,4)

env = gym.make('FrozenLake-v0', is_slippery=False)

obs = obs_to_torch(env.reset())
print('acabou de resetar: '+str(obs))
# Jogar manualmente
for _ in range(1000):
    env.render()
    #pi, value = neural(obs.unsqueeze(dim=0))
    for i in range(10):
        #print(pi)
        #a = pi.sample()
        #print(a)
        #a = int(a.cpu().numpy())
    	action = int(input("Digite uma ação entre: "+ str(env.action_space)))

    obs, reward, done, _ = env.step(action)
    print('b4 ' + str(obs))
    obs = obs_to_torch(obs)
    print('after: '+str(obs))


env.close()
