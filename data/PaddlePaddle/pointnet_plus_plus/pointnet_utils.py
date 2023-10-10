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
from __future__ import print_function
import paddle
from paddle.nn import Conv1D, ReLU, BatchNorm1D, Linear
import numpy as np
import paddle.nn.functional as F

class STN3d(paddle.nn.Layer):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = Conv1D(channel, 64, 1)
        self.conv2 = Conv1D(64, 128, 1)
        self.conv3 = Conv1D(128, 1024, 1)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 9)
        self.relu = ReLU()

        self.bn1 = BatchNorm1D(64)
        self.bn2 = BatchNorm1D(128)
        self.bn3 = BatchNorm1D(1024)
        self.bn4 = BatchNorm1D(512)
        self.bn5 = BatchNorm1D(256)


    def forward(self, x):
        batchsize = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = paddle.to_tensor(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).reshape((1,9)).tile((batchsize,1))
        # iden = iden.cuda()
        x = x + iden
        x = x.reshape((-1, 3, 3))
        return x


class STNkd(paddle.nn.Layer):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = Conv1D(k, 64, 1)
        self.conv2 = Conv1D(64, 128, 1)
        self.conv3 = Conv1D(128, 1024, 1)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k*k)
        self.relu = ReLU()

        self.bn1 = BatchNorm1D(64)
        self.bn2 = BatchNorm1D(128)
        self.bn3 = BatchNorm1D(1024)
        self.bn4 = BatchNorm1D(512)
        self.bn5 = BatchNorm1D(256)

        self.k = k

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = paddle.to_tensor(np.eye(self.k).flatten().astype(np.float32)).reshape((1, self.k*self.k)).tile((batchsize, 1))
        # iden = iden.cuda()
        x = x + iden
        x = x.reshape((-1, self.k, self.k))

        return x

class PointNetEncoder(paddle.nn.Layer):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = Conv1D(channel, 64, 1)
        self.conv2 = Conv1D(64, 128, 1)
        self.conv3 = Conv1D(128, 1024, 1)
        self.bn1 = BatchNorm1D(64)
        self.bn2 = BatchNorm1D(128)
        self.bn3 = BatchNorm1D(1024)
        self.relu = ReLU()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.shape
        trans = self.stn(x)
        x = x.transpose((0, 2, 1))
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = paddle.bmm(x, trans)
        if D > 3:
            x = paddle.concat((x, feature), axis=2)
        x = x.transpose((0, 2, 1))
        x = self.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose((0, 2, 1))
            x = paddle.bmm(x, trans_feat)
            x = x.transpose((0, 2, 1))
        else:
            trans_feat = None

        pointfeat = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = paddle.max(x, axis=2, keepdim=True)
        x = x.reshape((-1, 1024))
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.reshape((-1, 1024, 1)).tile((1, 1, N))
            return paddle.concat([x, pointfeat], axis=1), trans, trans_feat

class PointNetCls(paddle.nn.Layer):
    def __init__(self, k=40, feature_transform=False, normal_channel=False):
        super(PointNetCls, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform, channel=channel)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k)
        self.dropout = paddle.nn.Dropout(p=0.4)
        self.bn1 = BatchNorm1D(512)
        self.bn2 = BatchNorm1D(256)
        self.relu = ReLU()
        self.log_softmax = paddle.nn.LogSoftmax(axis=1)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return self.log_softmax(x), trans_feat

def feature_transform_regularizer(trans):
    d = trans.shape[1]
    batchsize = trans.shape[0]
    I = paddle.eye(d)[None, :, :]
    
    # I = I.cuda()
    loss = paddle.mean(F.normalize(paddle.bmm(trans, trans.transpose((2,1))) - I, axis=(1,2)))
    return loss

