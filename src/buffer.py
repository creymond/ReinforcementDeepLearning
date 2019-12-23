import random


class Memory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) > self.max_size:
            self.memory.pop(0)
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
