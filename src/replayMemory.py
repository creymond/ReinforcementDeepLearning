import random


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.index = 0
        self.memory = []

    def push(self, state, action, n_state, reward, done):
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = (state, action, n_state, reward, done)
        self.index = (self.index + 1) % self.capacity

    def pop(self, minibatch_size):
        return random.sample(self.memory, minibatch_size)

    def __len__(self):
        return len(self.memory)