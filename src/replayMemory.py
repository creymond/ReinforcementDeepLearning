import torch
from random import sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayMemory:
    def __init__(self, capacity, shape):
        self.capacity = capacity
        self.index = 0

        # Store each element independently
        # Gives the appropriate shape to tensor to store multidimensional data
        self.state = torch.empty([self.capacity, *shape], device=device)
        self.n_state = torch.empty([self.capacity, *shape], device=device)
        self.action = torch.empty([self.capacity], device=device)
        self.reward = torch.empty([self.capacity], device=device)
        self.done = torch.empty([self.capacity], device=device)

    def push(self, state, action, n_state, reward, done):
        i = self.index % self.capacity
        self.state[i] = torch.FloatTensor(state).to(device)
        self.n_state[i] = torch.FloatTensor(n_state).to(device)

        # Singleton must be converted as list before turning them to tensor
        self.action[i] = torch.LongTensor([action]).to(device)
        self.reward[i] = torch.FloatTensor([reward]).to(device)
        self.done[i] = torch.BoolTensor([done]).to(device)

        self.index = (self.index + 1)

    def pop(self, minibatch_size):
        list_index = []
        if self.index < self.capacity:
            for i in range(self.index):
                list_index.append(i)
        else:
            for i in range(self.capacity):
                list_index.append(i)
        s = sample(list_index, minibatch_size)

        # Get the values
        state = self.state[s, :]
        n_state = self.n_state[s, :]
        action = self.action[s]
        reward = self.reward[s]
        done = self.done[s]
        return state, action, n_state, reward, done
