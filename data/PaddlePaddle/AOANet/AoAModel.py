# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AoAModel.py
~~~~~~~~~~~

Implementation for paper 'Attention on Attention for Image Captioning'.
see https://arxiv.org/abs/1908.06954 for details.
"""
from collections import namedtuple

# paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# utils
from utils.utils import clones, attention

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']

BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq'])

class GLU(nn.Layer):
    """Applies the gated linear unit function."""
    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, input):
        return F.glu(input, axis=self.dim)

class SublayerConnection(nn.Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Layer):
    """Construct a layernorm module. See equation (7) in the paper for details."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = paddle.create_parameter(shape=(features, ), dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=1.))
        self.b_2 = paddle.create_parameter(shape=(features, ), dtype='float32',
                                           default_initializer=nn.initializer.Constant(value=0.))
        self.eps = eps

    def forward(self, x):
        mean = paddle.mean(x, axis=-1, keepdim=True)
        std = paddle.std(x, axis=-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MultiHeadAttention(nn.Layer):
    def __init__(self, h, d_model, dropout=0.1, project_k_v=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        """Inputs:
         - h: number of heads, default is 8.
         - d_model: dim of feature.
         - dropout: dropout prob for attention
         - project_k_v: do we need to do linear projections on K and V?
         - do_aoa: whether utilize aoa to refine the feature
         - norm_q: whether to norm the query
         - dropout_aoa: dropout prob of aoa
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume the dims of K and V are equal
        self.d_k = d_model // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x:x

        self.linears = clones(module=nn.Linear(d_model, d_model), N=1 + 2 * project_k_v)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x:x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        """Inputs:
         - query: [batch_size, num_objects, rnn_size]
         - value: [batch_size, num_objects, rnn_size]
         - key: [batch_size, num_objects, rnn_size]
         - masks: [batch_size, num_objects]
        """
        if mask is not None:
            if len(mask.shape) == 2:
                mask = paddle.unsqueeze(mask, axis=-2)
            # same mask applied to all h heads
            mask = paddle.unsqueeze(mask, axis=1)

        single_query = 0
        if len(query.shape) == 2:
            single_query = 1
            query = paddle.unsqueeze(query, axis=1)

        n_batch = query.shape[0]
        query = self.norm(query)

        # do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
            key_ = key.reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
            value_ = value.reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
        else:
            query_, key_, value_ = \
                [l(x).reshape((n_batch, -1, self.h, self.d_k)).transpose((0, 2, 1, 3))
                 for l, x in zip(self.linears, (query, key, value))]
        # apply attention on all the projected vectors in batch
        # x: [batch_size, num_heads, num_objects, rnn_size]
        # self.attn: [batch_size, num_heads, num_objects, num_objects]
        # see equation (8), (9), (10) in the paper for details
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)
        # concat
        # [batch_size, num_objects, rnn_size]
        x = x.transpose([0, 2, 1, 3]).reshape((n_batch, -1, self.h * self.d_k))

        if self.use_aoa:
            # apply AoA
            # see equation (6) for details
            x = self.aoa_layer(self.dropout_aoa(paddle.concat([x, query], axis=-1)))

        if single_query:
            query = paddle.squeeze(query, axis=1)
            x = paddle.squeeze(x, axis=1)
        return x

class AoA_Refiner_Layer(nn.Layer):
    def __init__(self, size, self_attn, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        return x

class AoA_Refiner_Core(nn.Layer):
    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadAttention(opt.num_heads, opt.rnn_size,
                                  project_k_v=1,
                                  do_aoa=1, norm_q=0,
                                  dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))
        layer = AoA_Refiner_Layer(opt.rnn_size, attn, 0.1)
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Inputs:
         - embeded_att_feats: [batch_size, num_objects, rnn_size]
         - mask: [batch_size, num_objects]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class AoA_Decoder_Core(nn.Layer):
    def __init__(self, opt):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        # AoA layer
        self.att2ctx = nn.Sequential(
            nn.Linear(self.d_model + opt.rnn_size, 2 * opt.rnn_size), GLU())

        self.attention = MultiHeadAttention(opt.num_heads, opt.rnn_size, project_k_v=0, do_aoa=0, norm_q=1)

        self.ctx_drop = nn.Dropout(self.drop_prob_lm)

    def forward(self, word_emb, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        """Inputs:
         - word_emb: [batch_size, input_encoding_size]
         - mean_feats: [batch_size, rnn_size]
         - att_feats: [batch_size, num_objects, rnn_size]
         - p_att_feats: [batch_size, num_objects, rnn_size * 2]
         - state: hidden state and memory cell of lstm.
         - att_mask: [batch_size, num_objects]
        """
        # state[0][1] is the context vector at the last step
        prev_h = state[0][1]

        # the input vector to the attention lstm consists of the previous output of lstm (prev_h),
        # mean-pooled image feature (mean_feats) and an encoding of the previous generated word (word_embs).
        # see equation (12) in the paper for details.
        att_lstm_input = paddle.concat([word_emb, mean_feats + self.ctx_drop(prev_h)], axis=1)
        _, (h_att, c_att) = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att,
                             paddle.slice(p_att_feats, axes=[2], starts=[0], ends=[self.d_model]),
                             paddle.slice(p_att_feats, axes=[2], starts=[self.d_model], ends=[self.d_model * 2]),
                             att_masks)
        ctx_input = paddle.concat([att, h_att], axis=1)
        output = self.att2ctx(ctx_input)

        # save the context vector to state[0][1]
        state = (paddle.concat([paddle.unsqueeze(h_att, 0), paddle.unsqueeze(output, 0)]),
                 paddle.concat([paddle.unsqueeze(c_att, 0), paddle.unsqueeze(state[1][1], 0)]))
        output = self.out_drop(output)
        return output, state

class AoAModel(nn.Layer):
    """Implementation of the attention on attention model."""
    def __init__(self, opt):
        super(AoAModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length # maximum sample length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.ss_prob = 0.0 # Schedule sampling probability

        # vocab_size + 1 for the <end> token
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))

        self.classifier = nn.Linear(self.rnn_size, self.vocab_size + 1)


        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]

        # mean pooling
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)

        self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)

        self.refiner = AoA_Refiner_Core(opt)
        self.core = AoA_Decoder_Core(opt)

    def _clip_att(self, att_feats, att_masks):
        """Clip the length of att_masks and att_feats to the maximum length

        Inputs:
         - att_feats: [batch_size, num_objects, att_dim]
         - att_masks: [batch_size, num_objects]
        """
        if att_masks is not None:
            max_len = paddle.cast(att_masks, dtype='int64').sum(axis=1).max()
            att_feats = att_feats[:, :max_len]
            att_masks = att_masks[:, :max_len]
        return att_feats, att_masks

    def _init_hidden(self, batch_size):
        """Init hidden state and cell memory for lstm."""
        return (paddle.zeros([self.num_layers, batch_size, self.rnn_size]),
                paddle.zeros([self.num_layers, batch_size, self.rnn_size]))

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        """Embed att_feats, and prepare att_feats for computing attention later.
        Inputs:
         - fc_feats: [batch_size, fc_feat_dim]
         - att_feats: [batch_size, num_objects, att_feat_dim]
         - att_masks: [batch_size, num_objects]
        """
        # embed att feats
        att_feats = self.att_embed(att_feats)
        scores_mask = paddle.full(shape=att_masks.shape, dtype=att_masks.dtype, fill_value=1e-9)
        scores = paddle.where(paddle.broadcast_to(att_masks, shape=scores_mask.shape) != 0, att_masks, scores_mask)
        att_feats *= paddle.unsqueeze(scores, axis=[-1])
        att_feats = self.refiner(att_feats, att_masks)

        # meaning pooling
        # use mean_feats instead of fc_feats
        if att_masks is None:
            mean_feats = paddle.mean(att_feats, axis=1)
        else:
            mean_feats = (paddle.sum(att_feats * paddle.unsqueeze(att_masks, axis=[-1]), axis=1) /
                          paddle.sum(paddle.unsqueeze(att_masks, axis=[-1]), axis=1))

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, att_masks

    def _forward_step(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        """Forward the LSTM each time step.

        Inputs:
         - it: previous generated words. paddle.LongTensor of shape [batch_size, ].
         - fc_feats: paddle.FloatTensor of shape [batch_size, rnn_size].
         - att_feats: paddle.FloatTensor of shape [batch_size, num_objects, rnn_size].
         - pre_att_feats: paddle.FloatTensor of shape [batch_size, num_objects, rnn_size * 2]
         - state: hidden state and memory cell of lstm.
        """
        word_embs = self.embed(it)
        output, state = self.core(word_embs, fc_feats, att_feats, p_att_feats, state, att_masks)
        output = self.classifier(output)

        logprobs = nn.functional.log_softmax(output, axis=1)
        return output, logprobs, state

    def forward(self, *inputs, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']

        # forward_xe or forward_rl depend on the mode
        return getattr(self, 'forward_' + mode)(*inputs, **kwargs)

    def forward_xe(self, fc_feats, att_feats, seq, att_masks=None):
        """Train the captioner with cross-entropy loss.

         - fc_feats: [batch_size, fc_feat_dim]
         - att_feats: [batch_size, num_objects, att_feat_dim]
         - seq: [batch_size, seq_len]
         - att_masks: [batch_size, num_objects]
        """
        batch_size = fc_feats.shape[0]

        # prepare feats
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # init lstm state
        state = self._init_hidden(batch_size)

        # logit_outputs = paddle.zeros((batch_size, seq.shape[1] -1, self.vocab_size + 1), dtype='float32')
        # prob_outputs = paddle.zeros((batch_size, seq.shape[1] -1, self.vocab_size + 1), dtype='float32')

        logit_outputs = []
        prob_outputs = []
        # this is because we add start and end token into the caption
        for i in range(seq.shape[1] - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                # using scheduled sampling
                sample_prob = paddle.uniform(shape=(batch_size, ), min=0, max=1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()  # [batch_size, ]
                else:
                    sample_ind = sample_mask.nonzero().reshape((-1,))
                    it = seq[:, i].clone()  # # [batch_size, ]
                    prob_prev = prob_outputs[i - 1].detach().exp()

                    index_selected = paddle.index_select(x=paddle.multinomial(prob_prev, num_samples=1).reshape((-1, )),
                                                         index=sample_ind, axis=0)

                    assert index_selected.shape[0] == sample_ind.shape[0]
                    # replace the groundtruth word with generated word when sampling next word
                    for j, ind in enumerate(sample_ind):
                        it[ind] = index_selected[j]
            else:
                it = seq[:, i].clone()

            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, logprobs, state = self._forward_step(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # used for compute loss
            logit_outputs.append(output)
            # used for sample words
            prob_outputs.append(logprobs)

        # we concat the output when finish all time steps
        logit_outputs = paddle.stack(logit_outputs, axis=1)
        prob_outputs = paddle.stack(prob_outputs, axis=1)  # [batch_size, max_len, vocab_size]

        return logit_outputs, prob_outputs

    def forward_sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)

        if beam_size > 1:
            batch_size = fc_feats.shape[0]
            seq = paddle.zeros((batch_size, self.seq_length), dtype='int64')
            seqLogprobs = paddle.zeros((batch_size, self.seq_length))

            for i in range(batch_size):
                seq_i, seqLogprobs_i = self._sample_beam(fc_feats[i], att_feats[i],
                                                         att_masks[i] if att_masks is not None else None,
                                                         opt)
                seq[i, :seq_i.shape[0]] = seq_i
                seqLogprobs[i, :seqLogprobs_i.shape[0]] = seqLogprobs_i

            return seq, seqLogprobs

        batch_size = fc_feats.shape[0]
        state = self._init_hidden(batch_size)

        # prepare feats
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # seq = paddle.zeros((batch_size, self.seq_length), dtype='int64')
        # seqLogprobs = paddle.zeros((batch_size, self.seq_length))
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length):
            if t == 0:  # input <bos>
                it = paddle.zeros((batch_size, ), dtype='int64')

            _, logprobs, state = self._forward_step(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break

            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished.cast('float32') * (it > 0).cast('float32')

            it = it * unfinished.cast(it.dtype)
            # seq[:, t] = it
            seq.append(it)
            # seqLogprobs[:, t] = sampleLogprobs.reshape((-1, ))
            seqLogprobs.append(sampleLogprobs.reshape((-1, )))

            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        # we concat the output when finish all time steps
        seq = paddle.stack(seq, axis=1).cast('int64')
        seqLogprobs = paddle.stack(seqLogprobs, axis=1)

        return seq, seqLogprobs

    def _sample_beam(self, fc_feat, att_feat, att_mask=None, opt={}):
        """Sampling with beam size."""
        decoding_constraint = opt.get('decoding_constraint', 0)
        beam_size = opt.get('beam_size')

        fc_feat = fc_feat.reshape((1, -1))  # (1, fc_feat_dim)
        att_feat = att_feat.reshape((1, -1, att_feat.shape[-1]))  # [1, num_objects, att_feat_dim]
        if att_mask is not None: att_mask = att_mask.reshape((1, -1))

        # prepare feats
        p_fc_feat, p_att_feat, pp_att_feat, p_att_mask = self._prepare_feature(fc_feat, att_feat, att_mask)
        # init state
        state = self._init_hidden(1)  # batch_size is 1

        candidates = [BeamCandidate(state, 0., [], 0, [])]  # 0 is for start and end
        for t in range(self.seq_length):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq = candidate
                if t > 0 and last_word_id == 0:  # end token
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    it = paddle.to_tensor([last_word_id], dtype='int64')
                    _, logprobs, state = self._forward_step(it, p_fc_feat, p_att_feat, pp_att_feat, p_att_mask,
                                                            state)
                    logprobs = logprobs.squeeze(0)  # [vocab_size, ]

                    # do not generate last step word
                    # i.e. do not sample same words between adjacent time step
                    if decoding_constraint:
                        logprobs[last_word_id] += float('-inf')

                    output_sorted = paddle.sort(logprobs, descending=True)
                    index_sorted = paddle.argsort(logprobs, descending=True)
                    # beam search
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(state, log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id, word_id_seq + [word_id]))

            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag: break

        _, _, log_prob_seq, _, word_id_seq = candidates[0]

        word_id_seq = paddle.to_tensor(word_id_seq, dtype='int64')
        log_prob_seq = paddle.to_tensor(log_prob_seq, dtype='float32')

        return word_id_seq, log_prob_seq

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs = paddle.max(logprobs, 1)
            it = paddle.argmax(logprobs, 1, True)
            it = it.reshape((-1, )).cast('int64')
        elif sample_method == 'sample':
            probs = paddle.exp(logprobs)
            it = paddle.multinomial(probs, 1)
            it = it.reshape((-1, )).cast('int64')
            # prepare data for paddle.gather_nd
            batch_size = it.shape[0]
            gather_index = paddle.zeros((batch_size, 2), dtype='int64')  # [batch_size, 2]
            gather_index[:, 0] = paddle.arange(batch_size)
            gather_index[:, 1] = it
            # gather the logprobs at sampled positions
            sampleLogprobs = paddle.gather_nd(logprobs, gather_index)

        return it, sampleLogprobs
