"""
Definição de conceitos:
Passo -> 1 movimento/ação do agente

Época -> Quantidade de passos do estado inicial até o final.
A quantidade de passos por época restringe o tamanho máximo de uma época, ou seja, se o agente não encontrar o seu objetivo a iteração é finalizada.

Batch -> Quantidade de épocas que serão utilizadas no treinamento.


"""
import time
from network import FeedForwardNN
from torch.distributions import Categorical
from mazefullaction import Maze

class PPO:
	#Inicializando o critico e o ator: A2C
	def __init__(self, env, max_steps, max_epochs):
		# Inicializando o ambiente
		self.env = env

		# Inicializando o ator
		self.actor = FeedForwardNN(env.get_obs_size() , env.get_action_size())
		# Inicializando o crítico
		self.critic = FeedForwardNN(env.get_obs_size(), 1)

		self.__init__hyperparameters(max_steps, max_epochs)


	def __init__hyperparameters(self, max_steps, max_epochs):
		# Quantidade de passos máximo de uma época
		self.max_steps = max_steps
		self.max_epochs = max_epochs

	def get_action(self, obs):
		# Obtendo a "média" das ações
		m = self.actor(obs)
		# Obtém a probabilidade de cada ação ocorrer
		dist = Categorical(logits=m)
		# Obtém a ação numa amostragem aleatória
		action = dist.sample()
		return action


	# Define cada época do treinamento, retorna a observação de todo o trajeto
	def step(self):
		# Conta a quantidade de passos feita pelo agente
		obs_array = []
		reward_array = []
		sum_reward_array = []

		epoch_count = 0
		
		while epoch_count < self.max_epochs:
			self.env.reset()
			step_count = 0
			sum_reward = 0
			# primeira iteração
			self.env.render()
			while step_count < self.max_steps: # Aqui estamos definindo o tamanho máximo de uma época, sendo max_steps
				time.sleep(1)
				obs_array_current = []
				obs_reward_current = []
				# Obtendo a observação do ambiente. Não se confunda com a obs obtida no openai gym
				# Obtendo uma possivel ação dado a observação do ambiente
				action = (self.get_action(self.env.get_obs()))
				print(str(action))
				obs, reward, terminate, _ = self.env.step(action) # Interagindo com o meio(environment)
				obs_array_current.append(obs)
				obs_reward_current.append(reward)
				sum_reward += reward
				if terminate == True:
					break
				step_count += 1
			obs_array.append(obs_array_current)
			reward_array.append(obs_reward_current)
			sum_reward_array.append(sum_reward)
			epoch_count += 1



	# Função que irá obter as observações obtidas em cada passo de treinamento
	def sampling(): 
		# Batch
		pass


# Just testing
if __name__ == "__main__":
	maze = Maze(8)
	ppo = PPO(maze, 100, 1)
	ppo.step()