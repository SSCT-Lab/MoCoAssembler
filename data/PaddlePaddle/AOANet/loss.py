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
loss.py
~~~~~~~

Loss functions used to compute loss.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils.utils import init_scorer, get_self_critical_reward

class RewardCriterion(nn.Layer):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.reshape((-1, ))
        reward = reward.reshape((-1, ))
        mask = paddle.to_tensor((seq > 0), dtype='float32')
        mask = paddle.concat(x=[paddle.full(shape=[mask.shape[0], 1], fill_value=1, dtype='float32'),
                                mask[:, :-1]], axis=1).reshape((-1, ))
        output = - input * reward * mask
        output = paddle.sum(output) / paddle.sum(mask)

        return output

class LabelSmoothing(nn.Layer):
    "Implement label smoothing."
    def __init__(self, vocab_size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.vocab_size = vocab_size + 1
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.shape[1]]
        mask = mask[:, :input.shape[1]]

        input = input.reshape((-1, input.shape[-1]))
        target = target.reshape((-1, ))
        mask = mask.reshape((-1, ))

        self.size = input.shape[1]
        target_one_hot = F.one_hot(target, num_classes=self.vocab_size)
        x = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.confidence)
        y = paddle.full(target_one_hot.shape, dtype=target_one_hot.dtype, fill_value=self.smoothing / (self.size - 1))
        true_dist = paddle.where(target_one_hot!=0, x, y)

        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()


class XECriterion(nn.Layer):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, mask):
        """Inputs:
         - pred: logits has shape of [batch_size, seq_len, vocab_size].
         - target: [batch_size, seq_len].
         - mask: [batch_size, seq_len].
        """
        # truncate to the same size
        target = target[:, :pred.shape[1]]
        mask = mask[:, :pred.shape[1]]

        loss_ = F.cross_entropy(pred, target, reduction='none')
        loss_ *= mask

        return paddle.sum(loss_) / paddle.sum(mask)

class LossWrapper(nn.Layer):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = LabelSmoothing(vocab_size=opt.vocab_size, smoothing=opt.label_smoothing)
        else:
            self.crit = XECriterion()
        self.rl_crit = RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices, sc_flag):
        out = {}
        if not sc_flag:
            self.model.train()
            logit_pred, prob_pred = self.model(fc_feats, att_feats, labels, att_masks)
            if self.opt.label_smoothing > 0:
                loss = self.crit(prob_pred, labels[:, 1:], masks[:, 1:])
            else:
                loss = self.crit(logit_pred, labels[:, 1:], masks[:, 1:])

        else:
            self.model.eval()
            with paddle.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method': 'sample'},
                                                     mode='sample')
            gts = [gts[_].numpy().astype('uint32') for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = paddle.to_tensor(reward, dtype='float32')
            loss = self.rl_crit(sample_logprobs, gen_result, reward)
            out['reward'] = reward[:, 0].mean()
        out['loss'] = loss
        return out
