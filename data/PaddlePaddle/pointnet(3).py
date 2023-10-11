import paddle
from paddle import nn


def conv1d_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv1D(in_dim, out_dim, 1),
        nn.BatchNorm1D(out_dim),
        nn.ReLU(),
    )


class BackBone(nn.Layer):
    def __init__(self, num_classes=16, n_points=2500):
        super().__init__()
        self.num_points = n_points
        self.num_classes = num_classes
        self.conv1 = nn.Conv1D(3, 64, 1)
        self.mlp1 = nn.Sequential(
            conv1d_bn_relu(3, 64),
            conv1d_bn_relu(64, 64),
        )
        self.mlp2 = nn.Sequential(
            conv1d_bn_relu(64, 128),
            conv1d_bn_relu(128, 1024),
        )

    def forward(self, x):
        """
        :param x: a array of shape [N, 3, n_points]
        :return: (low feature, high feature)
            low feature: a array of shape [N, 64, n_points]
            low feature: a array of shape [N, 1024, n_points]
        """
        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        return x1, x2


class PointNetClassifier(nn.Layer):
    def __init__(self, num_classes=16, n_points=2500):
        super().__init__()
        self.num_points = n_points
        self.num_classes = num_classes
        self.backbone = BackBone(self.num_classes, self.num_points)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
            nn.LogSoftmax(axis=-1)
        )

    def forward(self, x):
        x = paddle.transpose(x, (0, 2, 1))
        _, x = self.backbone(x)
        x = paddle.max(x, 2)  # [n, 1204]
        x = self.classifier(x)
        # print(x.shape)
        return x


class PointNetSeg(nn.Layer):
    def __init__(self, num_classes=6, n_points=2500):
        super().__init__()
        self.num_points = n_points
        self.num_classes = num_classes
        self.backbone = BackBone(self.num_classes, self.num_points)

        self.seg_head = nn.Sequential(
            conv1d_bn_relu(1024 + 64, 512),
            conv1d_bn_relu(512, 256),
            conv1d_bn_relu(256, 128),
            conv1d_bn_relu(128, self.num_classes),
            nn.LogSoftmax(axis=-1)
        )

    def forward(self, x):
        x = paddle.transpose(x, (0, 2, 1))
        l, h = self.backbone(x)  # low feature and high feature [n,64, 2500], [n, 1024, 2500]
        p = paddle.max(h, 2)  # [n, 1204] max pool
        p = paddle.unsqueeze(p, 2)  # [n, 1024, 1]
        p = paddle.tile(p, [1, 1, self.num_points])  # [n, 1024, 2500]
        x = paddle.concat([l, p], axis=1)  # [n, 1088, 2500]
        x = self.seg_head(x)  # [n, num_classes, 2500]
        x = paddle.transpose(x, (0, 2, 1))  # [n, 2500, num_classes]
        # print(x.shape)
        return x


if __name__ == '__main__':
    net = PointNetSeg()
    input_tensor = paddle.randn([16, 2500, 3])
    y = net(input_tensor)
    print(y.shape)