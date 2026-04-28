import torch
import torch.nn as nn

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.coeff = 1.0/6.0
    def forward(self, x:torch.Tensor):
        x = 1.2 * x + 3.0 # -2.5-2.5 -> 0-6.0
        x = self.relu6(x)
        x = x * self.coeff # to 0-1
        return x

class HardTanh(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.coeff = 2.0/6.0
    def forward(self, x:torch.Tensor):
        x = 3.0 * x + 3.0 # -1-1 -> 0-6
        x = self.relu6(x) # 0-6
        x = x * self.coeff - 1.0 # -1, 1
        return x

