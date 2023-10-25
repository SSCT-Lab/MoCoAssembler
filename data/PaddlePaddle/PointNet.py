import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TNet(nn.Layer):
    def __init__(self, k=64):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1D(k, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k
        self.iden = paddle.eye(self.k, self.k, dtype=paddle.float32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.reshape((-1, self.k, self.k)) + self.iden
        return x


class PointNetEncoder(nn.Layer):
    def __init__(
        self, global_feat=True, input_transform=True, feature_transform=False, channel=3
    ):
        super(PointNetEncoder, self).__init__()

        self.global_feat = global_feat
        if input_transform:
            self.input_transfrom = TNet(k=channel)
        else:
            self.input_transfrom = lambda x: paddle.eye(
                channel, channel, dtype=paddle.float32
            )

        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 64, 1)
        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(64)

        if feature_transform:
            self.feature_transform = TNet(k=64)
        else:
            self.feature_transform = lambda x: paddle.eye(64, 64, dtype=paddle.float32)

        self.conv3 = nn.Conv1D(64, 64, 1)
        self.conv4 = nn.Conv1D(64, 128, 1)
        self.conv5 = nn.Conv1D(128, 1024, 1)
        self.bn3 = nn.BatchNorm1D(64)
        self.bn4 = nn.BatchNorm1D(128)
        self.bn5 = nn.BatchNorm1D(1024)

    def forward(self, x):
        x = paddle.transpose(x, (0, 2, 1))
        B, D, N = x.shape
        trans_input = self.input_transfrom(x)
        x = paddle.transpose(x, (0, 2, 1))
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = paddle.bmm(x, trans_input)
        if D > 3:
            x = paddle.cat([x, feature], dim=2)
        x = paddle.transpose(x, (0, 2, 1))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        trans_feat = self.feature_transform(x)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.bmm(x, trans_feat)
        x = paddle.transpose(x, (0, 2, 1))

        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        if self.global_feat:
            return x, trans_input, trans_feat
        else:
            x = x.reshape((-1, 1024, 1)).repeat(1, 1, N)
            return paddle.cat([x, pointfeat], 1), trans_input, trans_feat


class PointNetClassifier(nn.Layer):
    def __init__(self, k=40, normal_channel=False):
        super(PointNetClassifier, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(
            global_feat=True,
            input_transform=True,
            feature_transform=True,
            channel=channel,
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans_input, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_input, trans_feat


def pointnet():
    return PointNetClassifier()

print(pointnet())
