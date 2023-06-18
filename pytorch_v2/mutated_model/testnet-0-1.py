import torch
import torch.nn as nn


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
    def forward(self, x):
        return x
def go():
    device = torch.device('cuda')
    net = testnet().to(device)
    x = torch.randn(3, 3, 224, 224).to(device)
    y = net(x)
