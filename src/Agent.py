import numpy as np
import torch
import torch.nn as nn

from network import DQN
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
    'N_EPOCHS': 300,
    'EPSILON': 0.9,
    'EPSILON_MIN': 0.0,
    'EPSILONE_DECAY': 0.992,
    'ALPHA': 0.005,
    'N_STEPS': 200,
    'N_ACTIONS': 2,
    'STEPS': 0
}


class Agent:
    def __init__(self, mem_size, hidden_dim, lr):
        self.dqn = DQN(hidden_dim)
        self.target_dqn = DQN(hidden_dim)
        self.eval_dqn = DQN(hidden_dim)

        self.conv = Convolutional()
        self.target_conv = Convolutional()
        self.eval_conv = Convolutional()

        self.memory = ReplayMemory(mem_size)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr)

        try:
            self.eval_dqn.load_state_dict(torch.load("Save/eval.data"))
            self.target_dqn.load_state_dict(torch.load("Save/eval.data"))
            print('Data loaded')
        except:
            pass

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
            q_value = self.eval_dqn(x)
            action = torch.argmax(q_value).item()
            return action
        else:
            action = random.randint(0, parameters['N_ACTIONS']-1)
            return action

    def actConv(self, state):
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
            q_value = self.eval_conv(x)
            action = torch.argmax(q_value).item()
            return action
        else:
            action = random.randint(0, parameters['N_ACTIONS']-1)
            return action

    def learn(self):
        if len(self.memory) < parameters['BATCH_SIZE']:
            return

        eval_dict = self.eval_dqn.state_dict()
        target_dict = self.eval_dqn.state_dict()

        for weights in eval_dict:
            target_dict[weights] = (1 - parameters['ALPHA']) * target_dict[weights] + parameters['ALPHA'] * eval_dict[
                weights]
            self.target_dqn.load_state_dict(target_dict)

        sample = self.memory.pop(parameters['BATCH_SIZE'])
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)

        l_batch_state = list(batch_state)
        te_batch_state = tuple(torch.from_numpy(e) for e in l_batch_state)

        loss = nn.MSELoss()
        for i in range(len(te_batch_state)):
            x = torch.FloatTensor(batch_state[i]).to(device)
            Q_current = self.eval_dqn(x)
            x_next = torch.FloatTensor(batch_next_state[i]).to(device)
            Q_next = self.target_dqn(x_next).detach()
            Q_expected = batch_reward[i] + (parameters['GAMMA'] * Q_next)
            l = loss(Q_current.max(), Q_expected.max())
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

    def learnConv(self):
        if len(self.memory) < parameters['BATCH_SIZE']:
            return

        eval_dict = self.eval_conv.state_dict()
        target_dict = self.eval_conv.state_dict()

        for weights in eval_dict:
            target_dict[weights] = (1 - parameters['ALPHA']) * target_dict[weights] + parameters['ALPHA'] * eval_dict[
                weights]
            self.target_conv.load_state_dict(target_dict)

        sample = self.memory.pop(parameters['BATCH_SIZE'])
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)

        # l_batch_state = list(batch_state)
        # print(type(l_batch_state[0]))
        # te_batch_state = tuple(torch.from_numpy(e) for e in l_batch_state)

        loss = nn.MSELoss()
        for i in range(len(batch_state)):
            x = torch.FloatTensor(batch_state[i]).to(device)
            Q_current = self.eval_conv(x)
            x_next = torch.FloatTensor(batch_next_state[i]).to(device)
            Q_next = self.target_conv(x_next).detach()
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
            action = env.action_space.pop(parameters['BATCH_SIZE'])
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
            rewards.append(0) # rewards per episode
            for s in range(parameters['N_STEPS']):
                action = self.act(state)
                n_state, reward, done, _ = env.step(action)
                if done:
                    reward = -1

                rewards[-1] += reward

                self.memory.push(state, action, n_state, reward, done)
                self.learn()

                state = n_state
                if i % 50 == 0:
                    torch.save(self.eval_dqn.state_dict(), "Model/eval.data")

            print('Epoch : ', i, ', Rewards : ', rewards[-1])

        torch.save(self.eval_dqn.state_dict(), "Model/eval.data")
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


    def atari(self):
        env = gym.make('BreakoutNoFrameskip-v4')
        env = AtariPreprocessing(env, scale_obs=True)
        env = FrameStack(env, 4)
        action_space = env.action_space.n
        rewards = []
        for episode in range(parameters['N_EPOCHS']):
            state = env.reset()
            rewards.append(0)
            done = False
            while not done:
                # env.render()
                action = self.actConv(state)
                n_state, reward, done, info = env.step(action)
                self.memory.push(state, action, n_state, reward, done)
                state = n_state
                rewards[-1] += reward
                self.learnConv()
            print("Episode : ", episode, ", Rewards : ", rewards[-1])
            if episode % 50 == 0:
                print("Saved !")
                torch.save(self.eval_conv.state_dict(), "Save/eval.data")

        torch.save(self.eval_conv.state_dict(), "Save/eval.data")

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
    agent = Agent(parameters['MEMORY_SIZE'], parameters['HIDDEN_DIM'], parameters['LEARNING_RATE'])
    # agent.dqn_cartpole()
    agent.atari()
