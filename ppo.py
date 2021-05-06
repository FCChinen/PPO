"""
Definição de conceitos:
Passo -> 1 movimento/ação do agente

Época -> Quantidade de passos do estado inicial até o final.
A quantidade de passos por época restringe o tamanho máximo de uma época, ou seja, se o agente não encontrar o seu objetivo a iteração é finalizada.

Batch -> Quantidade de épocas que serão utilizadas no treinamento.


"""

from network import FeedForwardNN

class PPO:
	#Inicializando o critico e o ator: A2C
	self __init__(self, env):
		# Quantidade de passos máximo de uma época
		self.max_steps = 1000
		self.actor = FeedForwardNN(env.get_obs_size() , env.get_action_size())
		self.critic = FeedForwardNN(env.get_obs_size(), 1)


	def get_action(self):


	# Define cada época do treinamento, retorna a observação de todo o trajeto
	def step(self):
		# primeira iteração
		for _ in range()
		obs = env.get_obs()

		action = self.get_action(obs)


	# Função que irá obter as observações obtidas em cada passo de treinamento
	def sampling(): 
		# Batch
