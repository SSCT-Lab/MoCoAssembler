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
import paddle.nn.functional as F
from time import time
import numpy as np
from models.pointnet_plus_plus_utils import PointNetSetAbstraction


class Pointnet2_cls_ssg(paddle.nn.Layer):
    def __init__(self,num_class,normal_channel=True):
        super(Pointnet2_cls_ssg, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = Linear(1024, 512)
        self.bn1 = BatchNorm1D(512)
        self.drop1 = paddle.nn.Dropout(p=0.4)
        self.fc2 = Linear(512, 256)
        self.bn2 = BatchNorm1D(256)
        self.drop2 = paddle.nn.Dropout(0.4)
        self.fc3 = Linear(256, num_class)
        self.log_softmax = paddle.nn.LogSoftmax(axis=-1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.reshape((B, 1024))
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = self.log_softmax(x)

        return x, l3_points

