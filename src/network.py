import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, hidden_dim):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x