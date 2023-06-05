import jittor
import jittor.nn as nn
# from jittorsummary import summary


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

    def execute(self, x):
        outputs = []
        layer = jittor.randn(x.shape[1], self.hidden_dim)
        for seq_lens in x:
            r = self.rx_linear(seq_lens) + self.rh_linear(layer)
            r = jittor.sigmoid(r)

            z = self.zx_linear(seq_lens) + self.zh_linear(layer)
            z = jittor.sigmoid(z)

            h_ = self.hx_linear(seq_lens) + self.hh_linear(r * layer)
            h_ = jittor.tanh(h_)

            layer = z * layer + (1 - z) * h_
            outputs.append(jittor.unsqueeze(layer, 0))

        x = jittor.cat(outputs)
        return x


if __name__ == '__main__':
    net = GRU(20, 100)
    x = jittor.randn((5, 100, 20))
    y = net(x)
    total_params = sum(p.numel() for p in net.parameters())
    print('GRU total params: ' + str(total_params))
    # summary(net, (100, 20))
