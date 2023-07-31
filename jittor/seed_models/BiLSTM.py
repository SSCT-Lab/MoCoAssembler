import jittor
import jittor.nn as nn


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm_forward = jittor.nn.LSTMCell(input_size=1, hidden_size=64)
        self.lstm_backward = jittor.nn.LSTMCell(input_size=1, hidden_size=64)
        self.fc = jittor.nn.Linear(in_features=128, out_features=1)

    def execute(self, x):
        batch_size, seq_length, _ = x.size()
        if True:
            h_forward = jittor.zeros((batch_size, 64))
            c_forward = jittor.zeros((batch_size, 64))
            h_backward = jittor.zeros((batch_size, 64))
            c_backward = jittor.zeros((batch_size, 64))
        outputs = []
        f = self.lstm_forward
        b = self.lstm_backward
        for i in range(seq_length):
            input_forward = x[:, i, :]
            input_backward = x[:, seq_length - i - 1, :]
            h_forward, c_forward = f(input_forward, (h_forward, c_forward))
            h_backward, c_backward = b(input_backward, (h_backward, c_backward))
            output = jittor.cat((h_forward, h_backward), dim=1)
            outputs.append(output)
        outputs = jittor.stack(outputs, dim=1)
        x = outputs
        x = self.fc(x)
        return x


def go():
    input_size = 1  # 输入维度
    hidden_size = 64  # LSTM隐藏层维度
    output_size = 1  # 输出维度
    model = BiLSTM()
    input_data = jittor.randn(5, 1, 1)
    model(input_data)
    return model
