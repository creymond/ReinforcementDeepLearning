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

parameters = {
    'GAMMA': 0.9,
    'MEMORY_SIZE': 2000,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-3,
    'N_EPISODE': 2000,
    'EPSILON': 0.9,
    'EPSILON_MIN': 0.1,
    'EPSILON_DECAY': 0.992,
    'ALPHA': 0.005,
    'N_STEPS': 200,
    'N_ACTIONS': 2,
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
            print("No data existing")

    def act(self, state):
        r = random.random()

        if r > self.p['EPSILON']:
            x = torch.FloatTensor(state)
            q_value = self.eval_cnn(x)
            action = torch.argmax(q_value).item()
            return action
        else:
            action = random.randint(0, self.p['N_ACTIONS'] - 1)
            return action

    def learn(self):
        if self.memory.index < self.p['BATCH_SIZE']:
            return

        # Get the state dict from the saved date
        eval_dict = self.eval_cnn.state_dict()
        target_dict = self.eval_cnn.state_dict()

        # Updating the parameters of the target DQN
        for w in eval_dict:
            target_dict[w] = (1 - self.p['ALPHA']) * target_dict[w] + self.p['ALPHA'] * eval_dict[w]
        self.target_cnn.load_state_dict(target_dict)

        # Get a sample of size BATCH
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.memory.pop(self.p['BATCH_SIZE'])

        # Update the treshold for the act() method if needed everytime the agent learn
        if self.p["EPSILON"] > self.p["EPSILON_MIN"]:
            self.p["EPSILON"] *= self.p["EPSILON_DECAY"]

        loss = nn.MSELoss()

        # Compute q values for the current evaluation
        q_eval = self.eval_cnn(batch_state).gather(1, batch_action.long().unsqueeze(1)).reshape([self.p["BATCH_SIZE"]])

        # Compute the next state q values
        q_next = self.target_cnn(batch_next_state).detach()

        # Compute the targetted q values
        q_target = batch_reward + q_next.max(1)[0].reshape([self.p["BATCH_SIZE"]]) * self.p["GAMMA"]
        self.optimizer.zero_grad()
        l = loss(q_eval, q_target)
        l.backward()
        self.optimizer.step()

    def atari(self):
        env = gym.make('BreakoutNoFrameskip-v4')
        env = env.unwrapped
        env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
        env = FrameStack(env, 4)
        rewards = []
        for i in range(self.p['N_EPISODE']):
            state = env.reset()
            rewards.append(0)
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
            print("Episode : ", i, ", Rewards : ", rewards[-1])

        torch.save(self.eval_cnn.state_dict(), "Model/eval_cnn.data")

        # Display result
        plt.ylabel("Rewards")
        plt.xlabel("Episode")
        plt.plot(rewards)
        plt.grid()
        plt.show()
        env.close()


if __name__ == "__main__":
    agent = AgentAtari(parameters)
    agent.atari()
