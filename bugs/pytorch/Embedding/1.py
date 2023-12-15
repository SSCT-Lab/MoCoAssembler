import torch
import torch.nn as nn


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.layer1 = torch.nn.LSTMCell(hidden_size=200, input_size=1, bias=False)
        self.layer2 = torch.nn.Embedding(num_embeddings=2, embedding_dim=2, padding_idx=3)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = self.layer2(x)
        return x


def go():
    model = lstm()
    x = torch.randn(5, 1)
    y = model(x)
    return model
