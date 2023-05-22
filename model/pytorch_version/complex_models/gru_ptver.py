from torch import nn
import torch


class GRU(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim

        # reset gate
        self.rx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.rh_linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # update gate
        self.zx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.zh_linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # tanh
        self.hx_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.hh_linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x):
        outputs = []
        layer = torch.randn(x.shape[1], self.hidden_dim)
        for seq_lens in x:
            r = self.rx_linear(seq_lens) + self.rh_linear(layer)
            r = torch.sigmoid(r)

            z = self.zx_linear(seq_lens) + self.zh_linear(layer)
            z = torch.sigmoid(z)

            h_ = self.hx_linear(seq_lens) + self.hh_linear(r * layer)
            h_ = torch.tanh(h_)

            layer = z * layer + (1 - z) * h_
            outputs.append(torch.unsqueeze(layer, 0))

        x = torch.cat(outputs)
        return x


if __name__ == '__main__':
    output = GRU(12, 6)(torch.randn(5, 24, 12))
    print(output.shape)
