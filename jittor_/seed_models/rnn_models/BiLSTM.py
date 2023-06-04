import jittor
import jittor as jt
import jittor.nn as nn
# from jittorsummary import summary


class BiLSTM(nn.Module):
    def __init__(self, batch_size, hidden_dim, vocab_size, sequence_len):
        super().__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = sequence_len

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim)

        # Bi-LSTM
        # Forward and backward
        self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_cell_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # LSTM layer
        self.lstm_cell = nn.LSTMCell(self.hidden_dim * 2, self.hidden_dim * 2)

        # Linear layer
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def execute(self, x):

        # Bi-LSTM
        # hs = [batch_size x hidden_size]
        # cs = [batch_size x hidden_size]
        hs_forward = jt.zeros(x.size(0), self.hidden_dim)
        cs_forward = jt.zeros(x.size(0), self.hidden_dim)
        hs_backward = jt.zeros(x.size(0), self.hidden_dim)
        cs_backward = jt.zeros(x.size(0), self.hidden_dim)

        # LSTM
        # hs = [batch_size x (hidden_size * 2)]
        # cs = [batch_size x (hidden_size * 2)]
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
        out = out.view(self.sequence_len, x.size(0), -1)

        forward = []
        backward = []

        # Unfolding Bi-LSTM
        # Forward
        for i in range(self.sequence_len):
            hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
            forward.append(hs_forward)

        # Backward
        for i in reversed(range(self.sequence_len)):
            hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)

        # LSTM
        for fwd, bwd in zip(forward, backward):
            input_tensor = jt.cat((fwd, bwd), 1)
            hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))

        # Last hidden state is passed through a linear layer
        x = self.linear(hs_lstm)

        return x


if __name__ == '__main__':
    HIDDEN_DIM = 100
    VOCAB_SIZE = 100
    SEQUENCE_LEN = 100
    net = BiLSTM(5, HIDDEN_DIM, VOCAB_SIZE, SEQUENCE_LEN)
    x = jittor.randn(100, 100)
    total_params = sum(p.numel() for p in net.parameters())
    y = net(x)
    print('total params: ' + str(total_params))
    # summary(net, (100,))
