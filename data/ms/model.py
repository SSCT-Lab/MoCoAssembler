# -*-coding:utf-8-*-
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
# ============================================================================

import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common.initializer import TruncatedNormal



class SharedMLP(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            pad_mode='valid',
            bn=False,
            activation_fn=None,
            bias=True
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.Conv2dTranspose if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode=pad_mode,
            has_bias=bias,
            weight_init=TruncatedNormal(sigma=1e-3)
        )
        self.has_bn = bn
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99)
        self.activation_fn = activation_fn

    def construct(self, x):
        r"""
            construct method

            Parameters
            ----------
            x: ms.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            ms.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(x)
        if self.has_bn:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Cell):
    def __init__(self, in_channel=10, out_channel=1, use_pos_encoding=True, bias=True):
        super(LocalSpatialEncoding, self).__init__()

        self.mlp = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias)
        self.d = out_channel
        self.use_pos_encoding = use_pos_encoding

    def construct(self, coords, features, neighbor_idx):
        r"""
            construct method

            Parameters
            ----------
            coords: ms.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: ms.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbor_idx: ms.Tensor, shape (B, N, K)

            Returns
            -------
            ms.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx = neighbor_idx

        # relative point position encoding
        cat = P.Concat(-3)
        if self.use_pos_encoding:
            # idx(B, N, K), coords(B, N, 3)
            # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
            extended_idx = P.Tile()(idx.expand_dims(1), (1, 3, 1, 1))
            extended_coords = P.Tile()(coords.transpose(0, 2, 1).expand_dims(-1),
                                       (1, 1, 1, idx.shape[-1]))
            neighbors = P.GatherD()(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)
            relative_coords = extended_coords - neighbors
            relative_dist = P.Sqrt()(P.ReduceSum(keep_dims=True)(P.Square()(relative_coords), -3))  # shape (B, 1, N, K)
            f_xyz = cat((
                relative_dist,
                relative_coords,
                extended_coords,
                neighbors
            ))
        else:
            f_xyz = coords

        f_xyz = self.mlp(f_xyz)  # (B,8,N,K)

        # (B, 8, N, 1) -> (B,8,N,K)
        f_tile = P.Tile()(features, (1, 1, 1, idx.shape[-1]))
        extended_idx_for_feat = P.Tile()(idx.expand_dims(1), (1, f_xyz.shape[1], 1, 1))
        f_neighbours = P.GatherD()(f_tile, 2, extended_idx_for_feat)  # (B,8,N,K) -> (B,8,N,K)

        # (B,8,N,K) & (B,8,N,K) -> (B,16,N,K)
        f_concat = cat([f_xyz, f_neighbours])

        if self.use_pos_encoding:
            return f_xyz, f_concat

        return f_concat


class AttentivePooling(nn.Cell):
    def __init__(self, in_channels, out_channels, bias):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.SequentialCell([
            nn.Dense(in_channels, in_channels, has_bias=False),
            nn.Softmax(-2)
        ])
        self.mlp = SharedMLP(in_channels, out_channels, bn=True,
                             activation_fn=nn.LeakyReLU(0.2), bias=bias)

    def construct(self, x):
        r"""
            construct method

            Parameters
            ----------
            x: ms.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            ms.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

        # sum over the neighbors
        features = scores * x
        features = P.ReduceSum(keep_dims=True)(features, -1)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Cell):
    def __init__(self, d_in, d_out, bias):
        super(LocalFeatureAggregation, self).__init__()

        self.mlp1 = SharedMLP(d_in, d_out//2, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias)
        self.mlp2 = SharedMLP(d_out, 2*d_out, bn=True, bias=bias)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True, bias=bias)

        self.lse1 = LocalSpatialEncoding(in_channel=10, out_channel=d_out//2, use_pos_encoding=True, bias=bias)
        self.lse2 = LocalSpatialEncoding(in_channel=d_out//2, out_channel=d_out//2, use_pos_encoding=False, bias=bias)

        self.pool1 = AttentivePooling(d_out, d_out//2, bias=bias)
        self.pool2 = AttentivePooling(d_out, d_out, bias=bias)

        self.lrelu = nn.LeakyReLU(0.2)

    def construct(self, coords, features, neighbor_idx):
        r"""
            construct method

            Parameters
            ----------
            coords: ms.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: ms.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbor_idx: ms.Tensor, shape (B, N, 16)
                knn neighbor idx
            Returns
            -------
            ms.Tensor, shape (B, 2*d_out, N, 1)
        """

        f_pc = self.mlp1(features)

        f_xyz, f_concat = self.lse1(coords, f_pc, neighbor_idx)
        f_pc_agg = self.pool1(f_concat)

        f_concat = self.lse2(f_xyz, f_pc_agg, neighbor_idx)
        f_pc_agg = self.pool2(f_concat)

        return self.lrelu(self.mlp2(f_pc_agg) + self.shortcut(features))


class RandLANet(nn.Cell):
    def __init__(self, d_in, num_classes, bias):
        super(RandLANet, self).__init__()

        self.fc_start = nn.Dense(d_in, 8)
        self.bn_start = nn.SequentialCell([
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ])

        # encoding layers
        self.encoder = nn.CellList([
            LocalFeatureAggregation(8, 16, bias=bias),
            LocalFeatureAggregation(32, 64, bias=bias),
            LocalFeatureAggregation(128, 128, bias=bias),
            LocalFeatureAggregation(256, 256, bias=bias),
            LocalFeatureAggregation(512, 512, bias=bias)
        ])

        self.mlp = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias)

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.LeakyReLU(0.2),
            bias=bias
        )
        self.decoder = nn.CellList([
            SharedMLP(1536, 512, **decoder_kwargs),
            SharedMLP(768, 256, **decoder_kwargs),
            SharedMLP(384, 128, **decoder_kwargs),
            SharedMLP(160, 32, **decoder_kwargs),
            SharedMLP(64, 32, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.SequentialCell([
            SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias),
            SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2), bias=bias),
            nn.Dropout(),
            SharedMLP(32, num_classes, bias=bias)
        ])

    def construct(self, xyz, feature, neighbor_idx, sub_idx, interp_idx):
        r"""
            construct method

            Parameters
            ----------
            xyz: list of ms.Tensor, shape (num_layer, B, N_layer, 3), each layer xyz
            feature: ms.Tensor, shape (B, N, d), input feature [xyz ; feature]
            neighbor_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer knn neighbor idx
            sub_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer pooling idx
            interp_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 1), each layer interp idx

            Returns
            -------
            ms.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """

        feature = self.fc_start(feature).swapaxes(-2, -1).expand_dims(-1)
        feature = self.bn_start(feature)  # shape (B, 8, N, 1)

        # <<<<<<<<<< ENCODER
        f_stack = []
        for i in range(5):
            # at iteration i, feature.shape = (B, d_layer, N_layer, 1)
            f_encoder_i = self.encoder[i](xyz[i], feature, neighbor_idx[i])
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                f_stack.append(f_encoder_i)
            f_stack.append(f_sampled_i)
        # # >>>>>>>>>> ENCODER

        feature = self.mlp(f_stack[-1])  # [B, d, N, 1]

        # <<<<<<<<<< DECODER
        f_decoder_list = []
        for j in range(5):
            f_interp_i = self.random_sample(
                feature, interp_idx[-j-1])  # [B, d, n, 1]
            cat = P.Concat(1)
            f_decoder_i = self.decoder[j](cat((f_stack[-j-2], f_interp_i)))
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # # >>>>>>>>>> DECODER

        scores = self.fc_end(f_decoder_list[-1])  # [B, num_classes, N, 1]

        return scores.squeeze(-1)

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, d, N', 1] pooled features matrix
        """
        b, d = feature.shape[:2]
        n_ = pool_idx.shape[1]
        # [B, N', max_num] --> [B, d, N', max_num]
        # pool_idx = P.repeat_elements(pool_idx.expand_dims(1), feature.shape[1], 1)
        pool_idx = P.Tile()(pool_idx.expand_dims(1),
                            (1, feature.shape[1], 1, 1))
        # [B, d, N', max_num] --> [B, d, N'*max_num]
        pool_idx = pool_idx.reshape((b, d, -1))
        pool_features = P.GatherD()(feature.squeeze(-1), -1, pool_idx)
        pool_features = pool_features.reshape((b, d, n_, -1))
        pool_features = P.ReduceMax(keep_dims=True)(
            pool_features, -1)  # [B, d, N', 1]
        return pool_features


class RandLAWithLoss(nn.Cell):
    """RadnLA-net with loss"""

    def __init__(self, network, weights, num_classes):
        super(RandLAWithLoss, self).__init__()
        self.network = network
        self.weights = weights
        self.num_classes = num_classes
        self.onehot = nn.OneHot(depth=num_classes)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, feature, labels, input_inds, cloud_inds,
                  p0, p1, p2, p3, p4, n0, n1, n2, n3, n4,
                  pl0, pl1, pl2, pl3, pl4, u0, u1, u2, u3, u4):
        xyz = [p0, p1, p2, p3, p4]
        neighbor_idx = [n0, n1, n2, n3, n4]
        sub_idx = [pl0, pl1, pl2, pl3, pl4]
        interp_idx = [u0, u1, u2, u3, u4]
        logits = self.network(xyz, feature, neighbor_idx, sub_idx, interp_idx)
        logit = logits.swapaxes(-2, -1).reshape((-1, self.num_classes))  # [b*n, 13]
        labels = labels.reshape((-1,))  # [b, n] --> [b*n]
        one_hot_labels = self.onehot(labels)  # [b*n, 13]
        weights = self.weights * one_hot_labels  # [b*n, 13]
        weights = P.ReduceSum()(weights, 1)  # [b*n]
        unweighted_loss = self.loss_fn(logit, one_hot_labels)  # [b*n]
        weighted_loss = unweighted_loss * weights  # [b*n]
        CE_loss = weighted_loss.mean()  # [1]

        return CE_loss


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]
