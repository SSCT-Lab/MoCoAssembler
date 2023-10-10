import os
import numpy as np
import paddle
import paddle.fluid as fluid


class InductionNetworks(object):
    """Induction Network core"""
    def __init__(self, pretrain_path, N, K, max_length, hidden_size,
                 att_dim, induction_iters, relation_size):
        """
        Args:
            pretrain_path: str. Path for word embedding and word id.
            N: int. N-way.
            K: int. K-shot.
            max_length: int. 
            hidden_size: int.
            att_dim: int.
            induction_iters: int.
            relation_size: int."""
        totalQ = fluid.data(name="totalQ", shape=[None], dtype="int32")  # total query
        total_Q = totalQ[0]
        support = fluid.data(name="support", shape=[None, N, K, max_length], dtype="int64")  # [B, N, K, T]
        support_len = fluid.data(name="support_len", shape=[None, N, K], dtype="int64")  # [B, N, K]
        query = fluid.data(name="query", shape=[None, None, max_length], dtype="int64")  # [B, totalQ, T]
        query_len = fluid.data(name="query_len", shape=[None, None], dtype="int64")  # [B, totalQ]
        label = fluid.data(name="label", shape=[None, None], dtype="int64")  # [B, totalQ]

        # Must be 3 data readers.
        # See https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/data_preparing/static_mode/use_py_reader.html
        self.train_reader = fluid.io.DataLoader.from_generator(
            feed_list=[totalQ, support, support_len, query, query_len, label],
            capacity=8, iterable=True
        )
        self.val_reader = fluid.io.DataLoader.from_generator(
            feed_list=[totalQ, support, support_len, query, query_len, label],
            capacity=8, iterable=True
        )
        self.test_reader = fluid.io.DataLoader.from_generator(
            feed_list=[totalQ, support, support_len, query, query_len, label],
            capacity=8, iterable=True
        )

        # 1. Encoder
        word_vec, vocab_size, embed_size = self.__load_embed_matrix(pretrain_path)

        support = fluid.layers.reshape(support, shape=[-1, max_length])  # [BNK, T]
        support_len = fluid.layers.reshape(support_len, shape=[-1])  # [BNK]
        support_emb = self.encoder_module(
            support, support_len, max_length, word_vec,
            vocab_size, embed_size, hidden_size, att_dim)  # [BNK, 2H]
        support_emb = fluid.layers.reshape(support_emb, shape=[-1, N, K, 2 * hidden_size])  # [B, N, K, 2H]

        query = fluid.layers.reshape(query, shape=[-1, max_length])  # [B*totalQ, T]
        query_len = fluid.layers.reshape(query_len, shape=[-1])  # [B*totalQ]
        query_emb = self.encoder_module(
            query, query_len, max_length, word_vec,
            vocab_size, embed_size, hidden_size, att_dim)  # [B*totalQ, 2H]
        query_emb = fluid.layers.reshape(query_emb, shape=[-1, total_Q, 2 * hidden_size])  # [B, totalQ, 2H]
        
        # 2. Induction
        class_emb = self.induction_module(support_emb, N, K, induction_iters,
                                          hidden_size)  # [B, N, 1, 2H]
        
        # 3. Relation
        relation_score = self.relation_module(class_emb, query_emb, N, total_Q,
                                              hidden_size, relation_size)  # [B, totalQ, N]
        
        # Return
        label_onehot = fluid.one_hot(label, depth=N)  # [B, totalQ, N]
        self.loss = fluid.layers.mse_loss(relation_score, label_onehot)  # [1]
        self.mean_acc = fluid.layers.accuracy(
            input=fluid.layers.reshape(relation_score, shape=[-1, N]),
            label=fluid.layers.reshape(label, shape=[-1, 1]))  # [1]
        self.prediction = fluid.layers.argmax(relation_score, axis=-1)  # [B, totalQ]
    
    def __load_embed_matrix(self, pretrain_path):
        # token2idx_dict = json.load(open(os.path.join(
        #     pretrain_path, "token2idx.json"), "r"))
        word_vec = np.load(os.path.join(
            pretrain_path, "word_vec.npy"))
        vocab_size, embed_size = word_vec.shape[0], word_vec.shape[1]
        # Unknown, Padding
        unk_idx, pad_idx = vocab_size, vocab_size + 1
        unk_vec = np.random.randn(1, embed_size) / np.sqrt(embed_size)
        pad_vec = np.zeros((1, embed_size), dtype="float32")
        word_vec = np.concatenate((word_vec, unk_vec, pad_vec), axis=0)
        return word_vec, vocab_size + 2, embed_size
    
    def encoder_module(self, seq_ids, seq_len, max_length, word_vec,
                       vocab_size, embed_size, hidden_size, att_dim):
        """Encoder module by Self-Attention Bi-LSTM."""
        # 1. Get embedding
        embed_param_attrs = fluid.ParamAttr(
            name="embed_weight",
            # learning_rate=0.5,
            initializer=fluid.initializer.NumpyArrayInitializer(word_vec),
            trainable=True)
        embed = fluid.embedding(
            input=seq_ids, size=[vocab_size, embed_size],
            padding_idx=-1, param_attr=embed_param_attrs)  # [-1, T, E]
        embed = fluid.layers.sequence_unpad(embed, length=seq_len)
        # LoDTensor. [-1*T', E], T' is true length of each sequecne
        
        # 2. BiLSTM layer
        fc_fw = fluid.layers.fc(input=embed, size=hidden_size * 4)
        fc_bw = fluid.layers.fc(input=embed, size=hidden_size * 4)

        lstm_h_fw, _ = fluid.layers.dynamic_lstm(
            input=fc_fw, size=hidden_size * 4, is_reverse=False)  # [-1*T', H]
        lstm_h_bw, _ = fluid.layers.dynamic_lstm(
            input=fc_bw, size=hidden_size * 4, is_reverse=True)  # [-1*T', H]

        pad_value = fluid.layers.assign(input=np.array([0.0], dtype="float32"))
        lstm_h_fw_padded, _ = fluid.layers.sequence_pad(
            lstm_h_fw, pad_value=pad_value,
            maxlen=max_length)  # [-1, T, H]
        lstm_h_bw_padded, _ = fluid.layers.sequence_pad(
            lstm_h_bw, pad_value=pad_value,
            maxlen=max_length)  # [-1, T, H]
        lstm_concat = fluid.layers.concat(input=[lstm_h_fw_padded, lstm_h_bw_padded], axis=-1)  # [-1, T, 2H]

        # 3. Attention layer
        att_score = fluid.layers.fc(input=lstm_concat, size=att_dim, num_flatten_dims=2,
                                    bias_attr=False, act="tanh")  # [-1, T, A]
        att_score = fluid.layers.fc(input=att_score, size=1, num_flatten_dims=2,
                                    bias_attr=False)  # [-1, T, 1]
        att_score = fluid.layers.softmax(input=att_score, axis=1)  # [-1, T, 1]
        lstm_att = fluid.layers.elementwise_mul(lstm_concat, att_score)  # [-1, T, 2H]
        lstm_att = fluid.layers.reduce_sum(lstm_att, dim=1)  # [-1, 2H]
        return lstm_att
    
    def induction_module(self, support_emb, N, K,
                         induction_iters, hidden_size):
        """Induction module by Dynamic Routing Algorithm."""
        def __squash(data):
            # data shape: [*, 2H]
            data_norm = fluid.layers.sqrt(
                fluid.layers.reduce_sum(fluid.layers.square(data),
                                        dim=-1, keep_dim=True))  # [*, 1]
            return fluid.layers.elementwise_div(
                fluid.layers.elementwise_mul(data, data_norm),
                fluid.layers.ones_like(data_norm) + fluid.layers.square(data_norm))
        
        # 1. Transform
        support_hat_emb = fluid.layers.fc(input=support_emb, size=2 * hidden_size,
                                          num_flatten_dims=3)  # [B, N, K, 2H]
        support_hat_emb = __squash(support_hat_emb)  # [B, N, K, 2H]

        # 2. Dynamic Routing iteration
        b = fluid.layers.zeros_like(support_hat_emb)  # [B, N, K, 2H]
        b = fluid.layers.reduce_sum(b, dim=-1)  # [B, N, K]
        b.stop_gradient = True
        for _ in range(induction_iters):
            d = fluid.layers.softmax(input=b, use_cudnn=True, axis=-1)  # [B, N, K]
            c_hat_emb = fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(support_hat_emb, d, axis=0),
                dim=2,
                keep_dim=True)  # [B, N, 1, 2H]
            c_emb = __squash(c_hat_emb)  # [B, N, 1, 2H]
            b = b + fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(support_hat_emb, c_emb),
                dim=-1)  # [B, N, K]
        return c_emb
    
    def relation_module(self, class_emb, query_emb, N, total_Q,
                        hidden_size, relation_size):
        """Relation module by 2 layers like Neural Tensor Network."""
        # 1. Adapt to shape
        query_emb = fluid.layers.unsqueeze(query_emb, axes=2)  # [B, totalQ, 1, 2H]
        query_emb = fluid.layers.expand(query_emb, expand_times=[1, 1, N, 1])  # [B, totalQ, N, 2H]
        query_emb = fluid.layers.reshape(query_emb, shape=[-1, 2 * hidden_size])  # [B*totalQ*N, 2H]

        class_emb = fluid.layers.transpose(class_emb, perm=[0, 2, 1, 3])  # [B, 1, N, 2H]
        class_emb = fluid.layers.expand(class_emb, expand_times=[1, total_Q, 1, 1])  # [B, totalQ, N, 2H]
        class_emb = fluid.layers.reshape(class_emb, shape=[-1, 2 * hidden_size])  # [B*totalQ*N, 2H]
        
        # 2. BiLinear layer
        v = fluid.layers.bilinear_tensor_product(class_emb, query_emb, size=relation_size,
                                                 act="relu", name="Relation-BiLinear")  # [B*totalQ*N, R]
        
        # 3. FC layer
        relation_score = fluid.layers.fc(input=v, size=1, act="sigmoid",
                                         name="Relation-FC")  # [B*totalQ*N, 1]
        relation_score = fluid.layers.reshape(relation_score, shape=[-1, total_Q, N])  # [B, totalQ, N]
        return relation_score