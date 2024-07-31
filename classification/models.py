
from torch import nn
import torch

class CircleModuleV0(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        # parameters:
        # 2*10 + 10*10 + 10*1 + 10 + 10 + 1
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return self.layer_3(torch.sigmoid(self.layer_2(torch.sigmoid(self.layer_1(x))))) # not work, split train/test half by half
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    