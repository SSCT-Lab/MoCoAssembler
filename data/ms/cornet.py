# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================================
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init

from mindspore import Parameter, Tensor, ms_function
from mindspore.common import dtype as mstype
import mindspore

ACT2FN = {'elu': ops.Elu(), 'relu': ops.ReLU(), 'sigmoid': ops.Sigmoid(), 'tanh': ops.Tanh()}

class CorNetBlock(nn.Cell):
    def __init__(self, context_size, output_size, cornet_act='sigmoid', **kwargs):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Dense(output_size, context_size)#.to_float(mstype.float16)
        self.cntxt2dstbn = nn.Dense(context_size, output_size)#.to_float(mstype.float16)
        self.act_fn = ACT2FN[cornet_act]
        self.elu = ACT2FN['elu']
        self.sigmoid = ops.Sigmoid()
        self.cast = ops.Cast()
    
    def construct(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.sigmoid(output_dstrbtn)
        # output_dstrbtn = self.cast(output_dstrbtn, mstype.float16)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        # context_vector = self.cast(context_vector, mstype.float32)

        context_vector = self.elu(context_vector)
        # context_vector = self.cast(context_vector,mstype.float16)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        # output_dstrbtn = self.cast(output_dstrbtn, mstype.float32)
        output_dstrbtn = output_dstrbtn + identity_logits        
        return output_dstrbtn
    
    
class CorNet(nn.Cell):
    def __init__(self, output_size, cornet_dim=1000, n_cornet_blocks=2, **kwargs):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.CellList([CorNetBlock(cornet_dim, output_size, **kwargs) for _ in range(n_cornet_blocks)])
        for layer in self.intlv_layers:
            # nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            # nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)
            init.XavierUniform(layer.dstbn2cntxt.weight)
            init.XavierUniform(layer.cntxt2dstbn.weight)

    def construct(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)        
        return logits

if __name__ == '__main__':

    import random
    import numpy as np

    # input = np.random.rand(6, 3, 32, 32)
    input = np.random.randint(low=1, high=10000, size=(40, 3801)).astype(np.float32)
    print(input.dtype)
    print(input.shape)
    input = Tensor(input)

    n1 = CorNet(output_size=3801)

    logits = n1(input)
    print("logits:",logits)
    # print(logits.shape)


