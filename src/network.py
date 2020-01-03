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


class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.fc1 = nn.Linear(5184, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x.reshape([-1, 4, 84, 84])))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
