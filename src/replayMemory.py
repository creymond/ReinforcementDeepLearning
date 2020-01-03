import random
import torch
from random import sample


class ReplayMemory:
    def __init__(self, capacity, shape):
        self.capacity = capacity
        self.index = 0

        # Contains the action, reward and boolean done
        self.memory = []

        # This allows the env to produce multidimensional state
        self.state = torch.empty([self.capacity, *shape])
        self.n_state = torch.empty([self.capacity, *shape])

    def push(self, state, action, n_state, reward, done):
        if len(self) < self.capacity:
            self.memory.append(None)

        # Adding action, reward, done as a tuple
        self.memory[self.index] = (action, reward, done)

        self.state[self.index] = torch.FloatTensor(state)
        self.n_state[self.index] = torch.FloatTensor(n_state)

        self.index = (self.index + 1) % self.capacity

    def pop(self, minibatch_size):
        list_index = []
        if len(self) < self.capacity:
            for i in range(self.index):
                list_index.append(i)
        else:
            for i in range(self.capacity):
                list_index.append(i)
        s = sample(list_index, minibatch_size)

        # Get the values
        state = self.state[s, :]
        n_state = self.n_state[s, :]
        action = []
        reward = []
        done = []
        for i in s:
            t = self.memory[i]
            action.append(t[0])
            reward.append(t[1])
            done.append(t[2])
        return state, action, n_state, reward, done

    def __len__(self):
        return len(self.memory)
