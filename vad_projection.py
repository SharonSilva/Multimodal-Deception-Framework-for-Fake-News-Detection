import torch
import torch.nn as nn

class VADProjection(nn.Module):
    def __init__(self, input_dim):
        super(VADProjection, self).__init__()
        self.linear = nn.Linear(input_dim, 3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.linear(x))  # (batch, 3)
