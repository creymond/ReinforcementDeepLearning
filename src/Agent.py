import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

from network import DQN
import matplotlib.pyplot as plt
import gym
from buffer import Memory
import random

cuda = False
use_cuda = torch.cuda.is_available() and cuda
device = torch.device("cuda" if use_cuda else "cpu")

parameters = {
    'HIDDEN_DIM': 256,
    'GAMMA': 0.8,
    'MEMORY_SIZE': 100000,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-3,
    'N_EPOCHS': 1000,
    'EPSILON': 0.9,
    'EPSILON_MIN': 0.05,
    'EPSILONE_DECAY': 200,
    'STEPS': 0,
    'N_ACTIONS': 2
}


class Agent:
    def __init__(self, mem_size, hidden_dim, lr):
        self.dqn = DQN(hidden_dim)
        self.memory = Memory(mem_size)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr)

    def act(self, state):
        r = random.random()
        e_min = parameters['EPSILON_MIN']
        e = parameters['EPSILON']
        e_decay = parameters['EPSILONE_DECAY']
        step = parameters['STEPS']
        epsilon_t = e_min + (e - e_min) * np.exp(-1. * step / e_decay)

        # Update step
        parameters['STEPS'] += 1

        if r > epsilon_t:
            x = torch.FloatTensor(state).to(device)
            q_value = self.dqn(x)
            action = torch.argmax(q_value).item()
            return action
        else:
            action = random.randint(0, parameters['N_ACTIONS']-1)
            return action

    def learn(self):
        if len(self.memory) < parameters['BATCH_SIZE']:
            return
        sample = self.memory.sample(parameters['BATCH_SIZE'])
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)

        l_batch_state = list(batch_state)
        te_batch_state = tuple(torch.from_numpy(e) for e in l_batch_state)

        loss = nn.MSELoss()
        for i in range(len(te_batch_state)):
            x = torch.FloatTensor(batch_state[i]).to(device)
            Q_current = self.dqn(x)
            x_next = torch.FloatTensor(batch_next_state[i]).to(device)
            Q_next = self.dqn(x_next).detach()
            Q_expected = batch_reward[i] + (parameters['GAMMA'] * Q_next)
            l = loss(Q_current.max(), Q_expected.max())
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

    def random(self):
        env = gym.make('CartPole-v1')

        env.reset()
        rewards = []
        while True:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break

        env.close()
        plt.ylabel("Rewards")
        plt.xlabel("Nb interactions")
        plt.plot(rewards)
        plt.grid()
        plt.show()


    def dqn_cartpole(self):
        env = gym.make('CartPole-v1')

        rewards = []
        for i in range(parameters['N_EPOCHS']):
            state = env.reset()
            parameters['STEPS'] = 0
            r = 0
            while True:
                env.render()
                action = agent.act(state)
                c_state = state
                n_state, reward, done, _ = env.step(action)
                if done:
                    reward = -1
                r += reward
                agent.memory.push(c_state, action, n_state, reward, done)

                agent.learn()


                state = n_state
                parameters['STEPS'] += 1
                if done:
                    print("Episode", i," finished after steps : ", parameters['STEPS'])
                    rewards.append(r)
                    break

        env.close()
        plt.ylabel("Rewards")
        plt.xlabel("Nb interactions")
        plt.plot(rewards)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    agent = Agent(parameters['MEMORY_SIZE'], parameters['HIDDEN_DIM'], parameters['LEARNING_RATE'])
    # agent.random()
    agent.dqn_cartpole()
