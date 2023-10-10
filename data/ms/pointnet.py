import mindspore.ops as ops
import mindspore.numpy as np
import mindspore as ms
from mindspore import nn
from models.blocks.t_net import STN3D, STNkD

from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class PointNetEncoder(nn.Cell):
    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3D()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.transpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        self.argmaxwithvalue = ops.ArgMaxWithValue(axis=2, keep_dims=True)
        self.reshape = ops.Reshape()

        self.relu = ops.ReLU()
        self.tile = ops.Tile()
        self.cat = ops.Concat(axis=1)

        if self.feature_transform:
            self.fstn = STNkD(k=64)

    def construct(self, x):  # 32 1024 3
        x = self.transpose(x, (0, 2, 1))
        transf = self.stn(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.batmatmul(x, transf)
        x = self.transpose(x, (0, 2, 1))
        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = self.transpose(x, (0, 2, 1))
            x = self.batmatmul(x, trans_feat)
            x = self.transpose(x, (0, 2, 1))
        else:
            trans_feat = None

        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1)))
        x = self.argmaxwithvalue(x)[1]

        if self.global_feat:
            x = self.reshape(x, (-1, 1024))
        return x, trans_feat


class PointNet_cls(nn.Cell):
    def __init__(self, k=40):
        super(PointNet_cls, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True)
        self.fc1 = nn.Dense(1024, 512)
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256, k)
        self.dropout = nn.Dropout(0.6)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(axis=1)

    def construct(self, x):
        x, _ = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = self.logsoftmax(x)
        return x  # 32 40


class PointNet_seg(nn.Cell):
    def __init__(self, part_num=50, normal_channel=True):
        super(PointNet_seg, self).__init__()
        if normal_channel:
            in_channel = 6
        else:
            in_channel = 3
        self.part_num = part_num
        self.stn = STN3D(in_channel)
        self.conv1 = nn.Conv1d(in_channel, 64, 1, has_bias=True, bias_init='normal')
        self.conv2 = nn.Conv1d(64, 128, 1, has_bias=True, bias_init='normal')
        self.conv3 = nn.Conv1d(128, 128, 1, has_bias=True, bias_init='normal')
        self.conv4 = nn.Conv1d(128, 512, 1, has_bias=True, bias_init='normal')
        self.conv5 = nn.Conv1d(512, 2048, 1, has_bias=True, bias_init='normal')
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(2048)
        self.fstn = STNkD(k=128)
        self.convs1 = nn.Conv1d(4944, 256, 1, has_bias=True, bias_init='normal')
        self.convs2 = nn.Conv1d(256, 256, 1, has_bias=True, bias_init='normal')
        self.convs3 = nn.Conv1d(256, 128, 1, has_bias=True, bias_init='normal')
        self.convs4 = nn.Conv1d(128, part_num, 1, has_bias=True, bias_init='normal')
        self.bns1 = nn.BatchNorm2d(256)
        self.bns2 = nn.BatchNorm2d(256)
        self.bns3 = nn.BatchNorm2d(128)

        self.transpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        self.cat1 = ops.Concat(axis=2)
        self.cat2 = ops.Concat(axis=1)
        self.relu = ops.ReLU()
        self.argmaxwithvalue = ops.ArgMaxWithValue(axis=2, keep_dims=True)
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.logsoftmax = nn.LogSoftmax(axis=-1)

    # def construct(self, x):
    #     feature = 0
    #
    #     x1 = x.transpose(0, 2, 1)
    #     # cls = self.transpose(x[:, :, -1:], (0, 2, 1))
    #     # = ops.Squeeze(-1)(cls[:, :, :1])  # 32 1
    #     cls = x1[:, :, -1:].squeeze(-1)
    #     # print(cls.shape)
    #     # print(cls)
    #     cls = cls[:, 0].astype(np.int32)
    #     one_hot = np.eye(16)
    #     label = one_hot[cls]
    #     # label = ops.Squeeze(-1)(self.transpose(label, (0, 2, 1)))  # 32 16
    #
    #     x = x1[:, :, :1024]  # 32 6 1024
    #     B, D, N = x.shape
    #     # x=x.transpose(0,2,1)
    #     trans = self.stn(x)
    #     point_cloud = self.transpose(x, (0, 2, 1))
    #     x = x.transpose(0, 2, 1)
    #     if D > 3:
    #         x = point_cloud[:, :, :3]
    #         feature = point_cloud[:, :, 3:]
    #     x = self.batmatmul(x, trans)
    #     if D > 3:
    #         x = self.cat1((x, feature))
    #
    #     x = self.transpose(x, (0, 2, 1))
    def construct(self, x, label):
        feature = 0
        x = self.transpose(x, (0, 2, 1))
        B, D, N = x.shape
        trans = self.stn(x)
        point_cloud = self.transpose(x, (0, 2, 1))
        if D > 3:
            x = point_cloud[:, :, :3]
            feature = point_cloud[:, :, 3:]
        x = self.batmatmul(x, trans)
        if D > 3:
            x = self.cat1((x, feature))

        x = self.transpose(x, (0, 2, 1))

        out1 = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        out2 = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(out1), -1))))
        out3 = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(out2), -1))))

        trans_feat = self.fstn(out3)
        x = self.transpose(out3, (0, 2, 1))
        net_transformed = self.batmatmul(x, trans_feat)
        net_transformed = self.transpose(net_transformed, (0, 2, 1))

        out4 = self.relu(ops.Squeeze(-1)(self.bn4(ops.ExpandDims()(self.conv4(net_transformed), -1))))
        out5 = ops.Squeeze(-1)(self.bn5(ops.ExpandDims()(self.conv5(out4), -1)))
        out_max = self.argmaxwithvalue(out5)[1]
        out_max = self.reshape(out_max, (-1, 2048))
        label = ops.Squeeze(1)(label)
        out_max = self.cat2((out_max, label))
        multiples = (1, 1, N)
        expand = self.tile(out_max.view(-1, 2048 + 16, 1), multiples)
        concat = self.cat2((expand, out1, out2, out3, out4, out5))
        net = self.relu(ops.Squeeze(-1)(self.bns1(ops.ExpandDims()(self.convs1(concat), -1))))
        net = self.relu(ops.Squeeze(-1)(self.bns2(ops.ExpandDims()(self.convs2(net), -1))))
        net = self.relu(ops.Squeeze(-1)(self.bns3(ops.ExpandDims()(self.convs3(net), -1))))
        net = self.convs4(net)
        net = self.transpose(net, (0, 2, 1))
        net = self.logsoftmax(net.view(-1, self.part_num))
        net = net.view(B, self.part_num, N)

        # return net, trans_feat
        return net

# x = ms.Tensor(np.ones((32,1024,6)),ms.float32)
# label = ms.Tensor(np.ones((32,16)),ms.float32)
# ms_model = PointNet_seg()
# net = ms_model(x,label)
# print(net.shape)
