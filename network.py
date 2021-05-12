import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

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