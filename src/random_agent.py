import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import wrappers, logger
from network import DQN
from buffer import Memory
import torch
import random
import torch.nn.functional as F



EPSILON = 0.5
BATCH_SIZE = 64
GAMMA = 0.8


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.memory = Memory(100000)
        self.dqn = DQN(256)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), 0.001)

    def act(self, state):
        r = random.random()
        if r > EPSILON:
            x = list(state)
            return self.dqn(torch.tensor(x))
        else:
            return torch.LongTensor([[random.randrange(2)]])

    def train(self):
        y = 0.01
        sample = self.memory.sample(np.min([10,len(self.memory)]))
        for i  in sample:
            x = self.dqn(torch.tensor(list(i[0])))
            xs = torch.max(self.dqn(torch.tensor(list(i[2]))))
            if i[4]==False:
                loss=F.mse_loss(x,(i[3]+y*xs)) #mse_Loss
                loss.backward()
            else:
                loss = F.mse_loss(x, torch.tensor(i[3] ))  # mse_Loss
                loss.backward()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    reward_list=[]
    m=buffer.Memory(100)
    #reward_list.append(reward)
    for i in range(episode_count):
        state = env.reset()
        while True:

            action = agent.act(state)
            action = torch.argmax(action).item()
            c_state = state
            n_state, reward, done, _ = env.step(action)
            agent.memory.push(c_state, action, n_state, reward, done)

            agent.train()

            reward_list.append(reward)
            if done:
                reward = -1

                break
    plt.plot(reward_list)

    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
