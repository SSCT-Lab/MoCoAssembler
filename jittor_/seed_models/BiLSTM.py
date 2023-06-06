import jittor
import jittor as jt
import jittor.nn as nn
# from jittorsummary import summary


class BiLSTM(nn.Module):
    def __init__(self, batch_size, hidden_dim, vocab_size, sequence_len):
        super().__init__()

        self.batch_size = 5
        self.hidden_dim = 100
        self.input_size = 100
        self.num_classes = 100
        self.sequence_len = 100
        self.dropout = jittor.nn.Dropout(0.25)
        self.embedding = jittor.nn.Embedding(100, 100)
        self.lstm_cell_forward = nn.LSTMCell(100, 100)
        self.lstm_cell_backward = nn.LSTMCell(100, 100)
        self.lstm_cell = nn.LSTMCell(200, 200)
        self.linear = nn.Linear(200, 100)

    def execute(self, x):

        # Bi-LSTM
        # hs = [batch_size x hidden_size]
        # cs = [batch_size x hidden_size]
        if True:
            hs_forward = jt.zeros(x.size(0), self.hidden_dim)
            cs_forward = jt.zeros(x.size(0), self.hidden_dim)
            hs_backward = jt.zeros(x.size(0), self.hidden_dim)
            cs_backward = jt.zeros(x.size(0), self.hidden_dim)

        # LSTM
        # hs = [batch_size x (hidden_size * 2)]
        # cs = [batch_size x (hidden_size * 2)]
        if True:
            hs_lstm = jt.zeros(x.size(0), self.hidden_dim * 2)
            cs_lstm = jt.zeros(x.size(0), self.hidden_dim * 2)

        # Weights initialization
        jt.nn.init.kaiming_normal_(hs_forward)
        jt.nn.init.kaiming_normal_(cs_forward)
        jt.nn.init.kaiming_normal_(hs_backward)
        jt.nn.init.kaiming_normal_(cs_backward)
        jt.nn.init.kaiming_normal_(hs_lstm)
        jt.nn.init.kaiming_normal_(cs_lstm)

        # From idx to embedding
        out = self.embedding(x.long())

        # Prepare the shape for LSTM Cells
        if True:
            out = out.view(self.sequence_len, x.size(0), -1)

        forward = []
        backward = []

        # Unfolding Bi-LSTM
        # Forward
        func = self.lstm_cell_forward
        for i in range(self.sequence_len):
            hs_forward, cs_forward = func(out[i], (hs_forward, cs_forward))
            forward.append(hs_forward)

        # Backward
        func = self.lstm_cell_backward
        for i in reversed(range(self.sequence_len)):
            hs_backward, cs_backward = func(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)

        # LSTM
        func = self.lstm_cell
        for fwd, bwd in zip(forward, backward):
            input_tensor = jt.cat((fwd, bwd), 1)
            hs_lstm, cs_lstm = func(input_tensor, (hs_lstm, cs_lstm))

        # Last hidden state is passed through a linear layer
        x = self.linear(hs_lstm)

        return x


if __name__ == '__main__':
    HIDDEN_DIM = 100
    VOCAB_SIZE = 100
    SEQUENCE_LEN = 100
    net = BiLSTM(5, HIDDEN_DIM, VOCAB_SIZE, SEQUENCE_LEN)
    x = jittor.randn(100, 100)
    # total_params = sum(p.numel() for p in net.parameters())
    y = net(x)
    # print('total params: ' + str(total_params))
    # summary(net, (100,))
