import numpy as np
import torch
import torch.nn as nn

from network import Convolutional
import matplotlib.pyplot as plt
import gym
from replayMemory import ReplayMemory
import random

from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack

cuda = False
use_cuda = torch.cuda.is_available() and cuda
device = torch.device("cuda" if use_cuda else "cpu")

parameters = {
    'HIDDEN_DIM': 30,
    'GAMMA': 0.9,
    'MEMORY_SIZE': 10000,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-3,
    'N_EPOCHS': 1,
    'EPSILON': 0.9,
    'EPSILON_MIN': 0.0,
    'EPSILONE_DECAY': 0.992,
    'ALPHA': 0.005,
    'N_STEPS': 200,
    'N_ACTIONS': 2,
    'STEPS': 0
}


class AgentAtari:
    def __init__(self, p):
        self.p = p

        self.target_cnn = Convolutional()
        self.eval_cnn = Convolutional()

        self.memory = ReplayMemory(self.p['MEMORY_SIZE'], [4, 84, 84])
        self.optimizer = torch.optim.Adam(self.eval_cnn.parameters(), self.p['LEARNING_RATE'])

        try:
            self.eval_cnn.load_state_dict(torch.load("Model/eval_cnn.data"))
            self.target_cnn.load_state_dict(torch.load("Model/eval_cnn.data"))
            print("Data has been loaded successfully")
        except:
            pass

    def act(self, state):
        r = random.random()
        e_min = self.p['EPSILON_MIN']
        e = self.p['EPSILON']
        e_decay = self.p['EPSILONE_DECAY']
        step = self.p['STEPS']
        epsilon_t = e_min + (e - e_min) * np.exp(-1. * step / e_decay)

        parameters['STEPS'] += 1

        if r > epsilon_t:
            x = torch.FloatTensor(state).to(device)
            q_value = self.eval_cnn(x)
            action = torch.argmax(q_value).item()
            return action
        else:
            action = random.randint(0, self.p['N_ACTIONS'] - 1)
            return action

    def learn(self):
        if self.memory.index < self.p['BATCH_SIZE']:
            return

        eval_dict = self.eval_cnn.state_dict()
        target_dict = self.eval_cnn.state_dict()

        for weights in eval_dict:
            target_dict[weights] = (1 - self.p['ALPHA']) * target_dict[weights] + self.p['ALPHA'] * eval_dict[
                weights]
            self.target_cnn.load_state_dict(target_dict)

        batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.memory.pop(self.p['BATCH_SIZE'])

        loss = nn.MSELoss()
        for i in range(len(batch_state)):
            x = torch.FloatTensor(batch_state[i]).to(device)
            Q_current = self.eval_cnn(x)
            x_next = torch.FloatTensor(batch_next_state[i]).to(device)
            Q_next = self.target_cnn(x_next).detach()
            Q_expected = batch_reward[i] + (self.p['GAMMA'] * Q_next)
            l = loss(Q_current.max(), Q_expected.max())
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

    def atari(self):
        env = gym.make('BreakoutNoFrameskip-v4')
        env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
        env = FrameStack(env, 4)
        rewards = []
        for episode in range(parameters['N_EPOCHS']):
            print('Episode ', episode)
            state = env.reset()
            rewards.append(0)
            done = False
            while not done:
                env.render()
                action = self.act(state)
                n_state, reward, done, info = env.step(action)
                self.memory.push(state, action, n_state, reward, done)
                state = n_state
                # When losing a life, it is not necessary to reduce the reward as it's not losing
                rewards[-1] += reward
                self.learn()
            print("Episode : ", episode, ", Rewards : ", rewards[-1])

        torch.save(self.eval_cnn.state_dict(), "Model/eval_cnn.data")

        state = env.reset()
        rewards.append(0)
        done = False
        while not done:
            env.render()
            action = self.act(state)
            observation, reward, done, info = env.step(action)
            rewards[-1] += reward
        print(rewards[-1])
        env.close()


if __name__ == "__main__":
    agent = AgentAtari(parameters)
    agent.atari()
