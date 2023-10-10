import paddle
import paddle.nn as nn
import numpy as np

from methods.meta_template import MetaTemplate
import paddle.nn.functional as F


class MatchingNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(MatchingNet, self).__init__(model_func, n_way, n_support)

        self.loss_fn = nn.NLLLoss()
        self.FCE = FullyContextualEmbedding(self.feat_dim)
        self.G_encoder = nn.LSTM(self.feat_dim, self.feat_dim, 1, direction='bidirectional')
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode_training_set(self, S, G_encoder=None):
        if G_encoder is None:
            G_encoder = self.G_encoder
        out_G = G_encoder(S.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = S + out_G[:, :S.shape[1]] + out_G[:, S.shape[1]:]
        G_norm = paddle.norm(G, p=2, axis=1).unsqueeze(1).expand_as(G)
        G_normalized = G.divide(G_norm + 0.00001)
        return G, G_normalized

    def get_logprobs(self, f, G, G_normalized, Y_S, FCE=None):
        if FCE is None:
            FCE = self.FCE
        F = FCE(f, G)
        F_norm = paddle.norm(F, p=2, axis=1).unsqueeze(1).expand_as(F)
        F_normalized = F.divide(F_norm + 0.00001)
        scores = self.relu(F_normalized.matmul(G_normalized, transpose_y=True)) * 100
        # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax
        softmax = self.softmax(scores)
        logprobs = (softmax.matmul(Y_S.cast('float32')) + 1e-6).log()
        return logprobs

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.reshape([self.n_way * self.n_support, -1])
        z_query = z_query.reshape([self.n_way * self.n_query, -1])
        G, G_normalized = self.encode_training_set(z_support)

        y_s = paddle.to_tensor(np.repeat(range(self.n_way), self.n_support))
        Y_S = F.one_hot(y_s, self.n_way)
        f = z_query
        logprobs = self.get_logprobs(f, G, G_normalized, Y_S)
        return logprobs

    def set_forward_loss(self, x):
        y_query = paddle.to_tensor(np.repeat(range(self.n_way), self.n_query))
        logprobs = self.set_forward(x)

        return self.loss_fn(logprobs, y_query.cast('int64'))

    def cuda(self):
        super(MatchingNet, self).cuda()
        self.FCE = self.FCE.cuda()
        return self


class FullyContextualEmbedding(nn.Layer):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.softmax = nn.Softmax()
        self.c_0 = paddle.zeros(shape=[1, feat_dim])
        self.feat_dim = feat_dim
        # self.K = K

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose([0, 1])
        K = G.shape[0]  # Tuna to be comfirmed
        for k in range(K):
            logit_a = h.matmul(G_T, transpose_y=True)
            a = self.softmax(logit_a)
            r = a.matmul(G)
            x = paddle.concat((f, r), 1)

            _, (h, c) = self.lstmcell(x, (h, c))
            h = h + f

        return h

    def cuda(self):
        super(FullyContextualEmbedding, self).cuda()
        self.c_0 = self.c_0.cuda()
        return self
