"""
Labirinto no qual o lagarto deve evitar os obstáculos(Árvores) até chegar no ponto final, a fim de tentar encontrar o melhor caminho.
O labirinto é uma matriz NxN, no qual N é passado por parâmetro pelo usuário, no qual N deve ser maior ou igual a 4. Pois senão, não é possivel adicionar obstáculos para obstruir a passagem do agente.


Modificações foram feitas nesse "custom environment" para se adequar ao formato dos ambientes do OpenAi GYM.
A versão original está no repositório QLearning

PS:. esse jogo não foi otimizado(modificações foram feitas sem levar em conta performance).

Esse labirinto é uma versão modificada do labirinto do repositório: https://github.com/FCChinen/QLearning

Autor: Felipe Churuyuki Chinen
"""

#import tkinter as tk
import gym
from gym import spaces
import math

class Maze(gym.Env):
    def __init__(self, pixels =40):
        #super(Maze, self).__init__()
        # Tamanho do labirinto
        self.N = 8
        # Tamanho de cada quadrado/posicao do labirinto, para imprimir a imagem
        self.pixels = pixels 

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(1)


        # INFORMAÇÕES DA LÓGICA DO JOGO
        # Soma das recompensas
        self.sum_reward = 0
        # Int que armazenará a observação/posicao do lagarto
        self.obs = (self.N-1)*(self.N-1)
        # Ultima posicao
        self.last_obs = (self.N-1)*(self.N-1)
        # posicao das arvores no ambiente
        self.trees = []
        # posicao dos grilos intermediarios
        self.inter_cricket = []
        # posicao do grilo final(Que leva ao final do jogo)
        self.final_cricket = []
        # array que contém os estados já percorridos pelo agente
        self.already_walked = [self.obs]
        # array que contém obstáculos
        self.obstacles = []

        self.create_trees()
        self.create_crickets()
        self.create_obstacles()


    def create_trees(self):
        trees = []
        for tree in range(0, self.N):
            trees.append(tree) # Criando árvores da borda inferior
            trees.append(tree + (self.N)*(self.N-1)) # Criando árvores da borda superior
            trees.append((self.N-1 + (self.N)*tree)) # Criando árvores da borda lateral direita
            trees.append(self.N * tree) # Criando árvores da borda lateral da esquerda
            #if (tree != self.N - 2):
            #    trees.append(tree + self.N*tree)
        self.trees = trees # Tirando as duplicatas

    def create_obstacles(self):
        # Adicionando arvores de obstaculo
        for i in range(2, 8, 2):
            if (i != 6):
                self.obstacles.append(self.tuple_to_int([i,1]))
                self.obstacles.append(self.tuple_to_int([i,3]))
                self.obstacles.append(self.tuple_to_int([i,5]))



    def create_crickets(self):
        #inter cricket
        #self.final_cricket.append(6 + 3*self.N)
        #final cricket
        self.final_cricket.append(self.N-2 + self.N)

    def render(self):
        for i in range(self.N*self.N):
            if i% (self.N) == 0:
                print()
            if i in self.trees:
                print("T", end='')
            elif i in self.final_cricket:
                print("C", end='')
            elif i == self.obs:
                print("L", end='')
            elif i in self.obstacles:
                print("X", end= '')
            else:
                print("0", end='')
        print('')

    def step(self, action):
        if (self.obs in self.trees):
            self.obs = self.last_obs
            rw = self.get_reward()
        else:
            self.last_obs = self.obs
            print('acao tomada: '+ str(action), 'na posicao:', str(self.obs))
            pos = self.get_next_pos(action)
            rw = self.get_reward()
        
        print("rw: "+str(self.get_reward()))

        return self.obs, rw, self.terminate(), {}
    
    def get_reward(self):
        if self.obs in self.trees:
            self.sum_reward -= 50
            return -50
        elif self.obs in self.final_cricket:
            self.sum_reward += 50
            return 50
        elif self.obs in self.inter_cricket:
            self.sum_reward += 5
            return 5
        elif self.obs in self.obstacles:
            self.sum_reward -= 5
            return -5
        else:
            """
            if self.obs in self.already_walked:
                self.sum_reward -= 1
                return -1
            else:
                self.sum_reward += 1
                return 1
            """
            return -1

            #self.sum_reward -=1
            #return -1

    def get_next_pos(self, action):
        if action == 0: # down
            self.obs += self.N
        elif action == 1: #left
            self.obs -= 1
        elif action == 2: # right
            self.obs += 1
        else: #up
            self.obs -= self.N
        return self.obs

    def terminate(self):
        #if self.obs in self.trees or self.obs in self.final_cricket:
        #    return True
        if self.obs in self.final_cricket:
            return True
        else:
            return False
    def reset(self):
        self.already_walked = []
        self.obs = (self.N-1)*(self.N-1)
        self.already_walked.append(self.obs)
        self.sum_reward = 0
        
        return self.obs

    def tuple_to_int(self, array):
        return array[0]+ self.N*array[1]

if __name__ == "__main__":
    maze = Maze(5)
    done = False
    obs = maze.reset()
    print("primeira obs:" + str(obs))
    while done == False:
        maze.render()
        action = int(input("Digite sua acao: 0 1 2 3: "))
        obs = maze.step(action)
        print("obs: "+ str(obs))
        done = maze.terminate()
