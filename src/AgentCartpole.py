import numpy as np
import torch
import torch.nn as nn

from network import DQN
import matplotlib.pyplot as plt
import gym
from replayMemory import ReplayMemory
import random

cuda = False
use_cuda = torch.cuda.is_available() and cuda
device = torch.device("cuda" if use_cuda else "cpu")

parameters = {
    'HIDDEN_DIM': 60,
    'GAMMA': 0.9,
    'MEMORY_SIZE': 10000,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-3,
    'N_EPOCHS': 10,
    'EPSILON': 0.9,
    'EPSILON_MIN': 0.0,
    'EPSILON_DECAY': 0.992,
    'ALPHA': 0.005,
    'N_STEPS': 200,
    'N_ACTIONS': 2,
    'STEPS': 0
}


class AgentCartpole:
    def __init__(self, p):
        self.p = p
        self.target_dqn = DQN(self.p['HIDDEN_DIM'])
        self.eval_dqn = DQN(self.p['HIDDEN_DIM'])

        self.memory = ReplayMemory(self.p['MEMORY_SIZE'], [4])
        self.optimizer = torch.optim.Adam(self.eval_dqn.parameters(), self.p['LEARNING_RATE'])

        try:
            self.eval_dqn.load_state_dict(torch.load("Model/eval_dqn.data"))
            self.target_dqn.load_state_dict(torch.load("Model/eval_dqn.data"))
            print("Data has been loaded successfully")
        except:
            pass

    def act(self, state):
        r = random.random()
        e_min = self.p['EPSILON_MIN']
        e = self.p['EPSILON']
        e_decay = self.p['EPSILON_DECAY']
        step = self.p['STEPS']
        epsilon_t = e_min + (e - e_min) * np.exp(-1. * step / e_decay)

        parameters['STEPS'] += 1

        if r > epsilon_t:
            x = torch.FloatTensor(state).to(device)
            q_value = self.eval_dqn(x)
            action = torch.argmax(q_value).item()
            return action
        else:
            action = random.randint(0, self.p['N_ACTIONS']-1)
            return action

    def learn(self):
        if self.memory.index < self.p['BATCH_SIZE']:
            return

        eval_dict = self.eval_dqn.state_dict()
        target_dict = self.eval_dqn.state_dict()

        for weights in eval_dict:
            target_dict[weights] = (1 - self.p['ALPHA']) * target_dict[weights] + self.p['ALPHA'] * eval_dict[
                weights]
            self.target_dqn.load_state_dict(target_dict)

        batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.memory.pop(self.p['BATCH_SIZE'])

        if self.p["EPSILON"] > self.p["EPSILON_MIN"]:
            self.p["EPSILON"] *= self.p["EPSILON_DECAY"]

        loss = nn.MSELoss()
        for i in range(len(batch_state)):
            x = torch.FloatTensor(batch_state[i]).to(device)
            Q_current = self.eval_dqn(x)
            x_next = torch.FloatTensor(batch_next_state[i]).to(device)
            Q_next = self.target_dqn(x_next).detach()
            Q_expected = batch_reward[i] + (self.p['GAMMA'] * Q_next)
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
            action = env.action_space.pop(self.p['BATCH_SIZE'])
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
        env = env.unwrapped
        rewards = []
        for i in range(self.p['N_EPOCHS']):
            state = env.reset()
            self.p['STEPS'] = 0
            rewards.append(0) # rewards per episode
            for s in range(self.p['N_STEPS']):
                env.render()
                action = self.act(state)
                n_state, reward, done, _ = env.step(action)
                if done:
                    reward = -1
                rewards[-1] += reward

                self.memory.push(state, action, n_state, reward, done)
                self.learn()

                state = n_state

            print('Epoch : ', i, ', Rewards : ', rewards[-1])

        torch.save(self.eval_dqn.state_dict(), "Model/eval_dqn.data")
        state = env.reset()
        done = False
        rewards.append(0)
        while not done:
            env.render()
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            rewards[-1] += reward
        plt.ylabel("Rewards")
        plt.xlabel("Epoch")
        plt.plot(rewards)
        plt.grid()
        plt.show()
        env.close()


if __name__ == "__main__":
    agent = AgentCartpole(parameters)
    agent.dqn_cartpole()
