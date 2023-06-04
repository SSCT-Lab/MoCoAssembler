import jittor
import jittor.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = jittor.nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = jittor.nn.Linear(hidden_size, output_size)

    def execute(self, x) -> None:
        h0 = jittor.zeros(self.num_layers, x.shape[1], self.hidden_size)
        c0 = jittor.zeros(self.num_layers, x.shape[1], self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1, :, :])
        return out


if __name__ == '__main__':
    HIDDEN_SIZE = 200
    NUM_LAYERS = 4
    OUTPUT_SIZE = 20
    x = jittor.randn(10, 10)
    net = LSTM(10, HIDDEN_SIZE, NUM_LAYERS-2, OUTPUT_SIZE)
    total_params = sum(p.numel() for p in net.parameters())
    y = net(x)
    print('total_params: ' + str(total_params))
    print(net)
