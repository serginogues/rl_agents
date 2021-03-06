import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    """
    Defines a deep Q network with a single hidden layer
    """
    def __init__(self, learning_rate, n_actions, input_dims, checkpoint_dir, name):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.to(self.device)
        print(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(checkpoint_dir, name)

    def forward(self, data):
        fc_layer1 = F.relu(self.fc1(data))
        fc_layer2 = F.relu(self.fc2(fc_layer1))
        actions = self.fc3(fc_layer2)

        return actions

    def save_checkpoint(self):
        print('Saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_name)

    def load_checkpoint(self):
        print('Loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_name))


