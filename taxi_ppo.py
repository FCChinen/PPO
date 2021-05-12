# Importing the custom env
from mazefullaction import Maze

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.distributions import Categorical
import time
from typing import Dict, List
import gym

# Definindo que a cpu ira processar tudo
device = torch.device("cpu")

# Funcao define a rede neural
class FeedForwardNN(nn.Module):
    def __init__(self, in_dim=1, out_dim=4):
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


# Funcao que transforma np.ndarray para o tipo Tensor
# Tambem transforma os valores da observacao no intervalo [0,1]
def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class PPO:
    def __init__(self, env, max_steps = 100):
     #   self.__init__hyperparameters(max_steps = 100)
        # Variavel que armazena o ambiente
        self.env = env

        # Hyper-paraemtros
        # Numero maximo de passos numa epoca
        self.max_steps = max_steps
        # 
        self.gamma = 0.95
        # Quantidade de epocas
        self.epochs = 4
        # Tamanho de uma batch(Esse tamanho ditara a quantidade de amostras para fazer uma atualizacao da politica)
        self.batch_size = self.max_steps
        # Tamanho de uma mini batch
        self.mini_batch_size = 4
        #
        self.updates = 1000

        # Armazena a soma de todas as recompensas ate o ponto T que se encontra
        self.sum_rewards = 0
        self.obs = []

        # Criando o modelo da rede neural
        self.model = FeedForwardNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

    #def __init__hyperparameters(self, max_steps):
         # Quantidade maxima passos numa época


    def step(self, action):
        """
        Essa funcao descreve cada passo do treinamento.
        """
        obs, reward, done, _ = self.env.step(action)

        #self.sum_rewards.append(reward)

        return obs, reward, done

    def reset(self):
        self.obs = self.env.reset()
        self.sum_rewards = [] # Reseta o array que contem as recompensas acumuladas

        return self.obs

    def train(self):
        pass

    def _calc_advantages(self, done : np.ndarray, rewards : np.ndarray, values : np.ndarray) -> np.ndarray:
        advantages = np.zeros((self.max_steps), dtype=np.float32)
        last_advantage = 0

        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data

        for t in reversed(range(self.max_steps)):
            mask = 1.0 - done[t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[t] + self.gamma + last_value - values[t]

            last_value = values[t]

        return advantages

    def sample(self):
        # Contém as amostras de cada época
        rewards_array = np.zeros((self.max_steps), dtype=np.int32)
        actions_array = np.zeros((self.max_steps), dtype=np.int32)
        done_array = np.zeros((self.max_steps), dtype=np.bool)
        obs_array = np.zeros((self.max_steps, 64), dtype=np.int32) # 64 é o numero de posicoes que o agente pode navegar
        log_pis_array = np.zeros((self.max_steps), dtype=np.float32) # log da politica
        values_array = np.zeros((self.max_steps), dtype=np.float32) # value function

        obs = self.reset() # Armazenando a ultima observacao
        # Cada passo do treinamento

        for t in range(self.max_steps): 
            with torch.no_grad():
                obs_array[t] = obs

                obs = torch.unsqueeze(obs_to_torch(obs), dim=0)

                pi, v = self.model(obs)
                
                values_array[t] = v.cpu().numpy()
                a = pi.sample()
                print(a)
                actions_array[t] = a.cpu().numpy()
                log_pis_array[t] = pi.log_prob(a).cpu().numpy()
                # Obtendo a informacoes do passo, dado a acao a.
                new_obs, new_reward, new_done = self.step(a)

                obs_array[t,:] = new_obs
                rewards_array[t] = new_reward
                done_array[t] = new_done

                if new_done == True:
                    break


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
            #v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)
        print(len(obs_array))
        return samples_flat

    def train(self, samples: Dict[str, torch.Tensor], learning_rate: float, clip_range: float):
        for _ in range(self.epochs):
            # Obtendo o index das amostras de maneira aleatória
            idx = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_idx = idx[start : end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_idx]

                loss = self._calc_loss(clip_range = clip_range, samples=mini_batch)

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
        sampled_return = samples['values'] + samples['advantage'] # 

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

            #torch.save(self.model.state_dict(), './model.pth')

    def test_loop(self, number_it):
        for i in range(number_it):
            obs = obs_to_torch(self.env.reset())
            done = False
            while done == False:
                pi, value = self.model(obs)
                action = pi.sample()
                obs, reward, done, _ = self.env.step(action)
                obs = obs = obs_to_torch(obs)
                time.sleep(1)
                print(str(done))


if __name__ == '__main__':
    maze = Maze(8)
    env = gym.make('FrozenLake8x8-v0')
    ppo = PPO(env)
    ppo.run_training_loop()
    print('terminou treinamento')
    ppo.test_loop(100)