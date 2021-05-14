"""
0 - Left
1 - Down
2 - Right
3 - Up
"""

import gym
import torch
import numpy as np
from network import FeedForwardNN
from typing import Dict, List
from torch import optim
import json
import time
from newmaze import Maze
device = torch.device("cpu")

# Transforma um np.array pra um tensor
# E normaliza a entrada para um valor entre 0,1
def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device) / 64.


class PPO:
    def __init__(self, env, max_steps = 10000):

        self.env = env

        #Hyper Parametros
        self.max_steps = max_steps
        # Fator de desconto
        self.gamma = 0.99
        self.lamda = 0.95

        # Quantidade de épocas
        self.epochs = 10
        # Quantidade de passos que tera uma batch
        self.batch_size = self.max_steps
        # Tamanho de uma mini batch !
        self.mini_batch_size = self.batch_size//100
        # quantidade de atualizações que ocorrerá na politica
        self.updates = 10

        self.max_iterations = 16

        # Soma das recompensas
        self.sum_rewards = 0
        self.obs = [] # !
        
        
        # Inicializa o modelo da rede neural
        self.model = FeedForwardNN(in_dim = 1, out_dim = 4, hidden_layer = 16)

        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

        # Array que contem a loss
        self.loss = []
    def step(self, action):

        obs, reward, done = self.env.step(action)
        obs = obs_to_torch(obs)
        obs = obs.unsqueeze(dim=0) # Transforma a entrada de tensor(x) tensor([x]), basicamente um tratamento para utilizar na rede neural
        return obs, reward, done

    def reset(self):
        obs = self.env.reset()
        obs = obs_to_torch(obs)
        obs = obs.unsqueeze(dim=0) # Transforma a entrada de tensor(x) tensor([x]), basicamente um tratamento para utilizar na rede neural
        self.sum_rewards = []
        return obs

    def _calc_advantages(self, done : np.ndarray, rewards : np.ndarray, values : np.ndarray) -> np.ndarray:
        # Essa função calcula as vantagens
        # Voltar nessa função quando ela aparecer no código
        advantages = np.zeros((self.max_steps), dtype=np.float32)
        last_advantage = 0.

        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data
        
        # Calcula de ordem reversa Não sei explicar o porque de fato
        for t in reversed(range(self.max_steps)):
            mask = 1.0 - done[t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[t] + self.gamma + last_value - values[t]

            last_advantage = delta + self.gamma * self.lamda * last_advantage

            advantages[t] = last_advantage

            last_value = values[t]            

        return advantages

    def sample(self):
        rewards_array = []
        actions_array = []
        done_array = []
        obs_array = []
        log_pis_array = []
        values_array = []

        ep_qty = 100 # Quantidade de épocas
        ep_count = 0
        while ep_count < ep_qty:
            rewards_cur = []
            actions_cur = []
            done_cur = []
            obs_cur = []
            log_pis_cur = []
            values_cur = []

            self.obs = self.reset()
            finished_ep = False
            steps_count = 0
            while ((finished_ep == True) or (steps_count < 50)):
                with torch.no_grad():
                    obs_cur.append(self.obs)

                    pi, v = self.model(self.obs)

                    values_cur.append(v.cpu().numpy())
                    a = pi.sample()
                    action=int(a.cpu().numpy())
                    log_pis_cur.append(pi.log_prob(a).cpu().numpy)

                    self.obs, new_reward, new_done = self.step(action)

                    rewards_cur.append(new_reward)
                    done_cur.append(new_done)
                    finished_ep = new_done

                    if new_done == True:
                        rewards_array.append(rewards_cur)
                        actions_array.append(actions_cur)
                        done_array.append(done_cur)
                        obs_array.append(obs_cur)
                        log_pis_array.append(log_pis_cur)
                        values_array.append(values_cur)
                        

        # Calcula a vantagem(Generalized Advantage Estimator)
        advantage = self._calc_advantages(done_array, rewards_array, values_array) 
        

        # Cria um dicionario que contem as informacoes de cada amostra
        samples = {
            'obs' : obs_array,
            'actions' : actions_array,
            'values' : values_array,
            'log_pis' : log_pis_array,
            'advantage' : advantage
        }

        # Cria um dicionario que contera as amostras
        samples_flat = {}
        for k, v in samples.items():
            #import pdb; breakpoint()
            v = v.reshape(-1, 1)
            if k == 'obs':
                #samples_flat[k] = torch.unsqueeze(obs_to_torch(v),0)
                samples_flat[k] = (obs_to_torch(v))
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat
    
    def train(self, samples: Dict[str, torch.Tensor], learning_rate: float, clip_range: float):
        for _ in range(self.epochs):
            # Obtendo o index das amostras de maneira aleatória
            idx = torch.randperm(self.batch_size)
            # Está criando o loop que irá descrever o processo de atualização da politica
            # o mini_batch dita a quantidade de observacoes por batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size 
                mini_batch_idx = idx[start : end]
                mini_batch = {}
                
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_idx]

                loss = self._calc_loss(clip_range = clip_range, samples=mini_batch)
                
                self.loss.append(loss)

                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8) # Normalizando, foi adicionado 1e-8 para garantir que não haja divisao por zero


    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range : float) -> torch.Tensor:
        # sample['values'] and samples['advantage'] sao logs
        # queremos calcular pi_new / pi_old
        # como são logs, utilizamos a seguinte propriedade
        # log(a-b) = log a / log b
        sampled_return = samples['values'] + samples['advantage'] 

        sampled_normalized_advantage = self._normalize(samples['advantage']) # normalization <- is it actually needed? !

        pi, value = self.model(samples['obs']) # retreaving information about the model, the policy and the value-fuction

        log_pi = pi.log_prob(samples['actions']) # applying log to the probility !

        ratio = torch.exp(log_pi - samples['log_pis']) # new_policy - old_policy

        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range) # returning the clipped ratio

        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage) # utilizing the formula on PPO article
        policy_reward = policy_reward.mean() # ! why the mean?
        
        entropy_bonus = pi.entropy() # ! 
        entropy_bonus = entropy_bonus.mean() # !

        # Value of the observation - clipped estimated value(of the neural network)
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range,
                                                                              max=clip_range)

        # value function loss, getting the maximun value
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)

        # ! why the mean of 1/2 * mean of loss?
        vf_loss = 0.5 * vf_loss.mean()

        #   multiplying for -1, thats why they get the maximun value on vf_loss 2 lines above
        loss = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        # kl_divergence
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()

        # 
        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()

        return loss


    def run_training_loop(self):
        for update in range(self.updates):
            progress = update/self.updates
            learning_rate = 2.5e-4 * (1-progress)
            clip_range = 0.1 * (1-progress)
            
            samples = self.sample()
            
            self.train(samples, learning_rate, clip_range)
            print('chegou ate aqui')


    def test_loop(self, number_it):
        for i in range(number_it):
            obs = obs_to_torch(self.env.reset())
            done = False
            while done == False:
                env.render()
                pi, value = self.model(obs.reshape(1, -1))
                action = pi.sample()
                obs, reward, done = self.env.step(int(action))
                obs = obs_to_torch(obs)
                time.sleep(1)
                if done == True:
                    print("GG WP")

if __name__ == "__main__":
    env = Maze(5)
    ppo = PPO(env)
    sample = ppo.run_training_loop()
    ppo.test_loop(1)


"""
FrozenLake8x8-v0
The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend.

The surface is described using a grid like the following:

SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.

VIEW SOURCE ON GITHUB
"""
