import paddle
from paddle.io import Dataset
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear, BatchNorm, Dropout, ReLU, Softmax, Sequential


class PointNet(paddle.nn.Layer):
    def __init__(self, name_scope='PointNet_', num_classes=4, num_point=1024):
        super(PointNet, self).__init__()
        self.num_point = num_point
        self.input_transform_net = Sequential(
            Conv2D(1, 64, (1, 3)),
            BatchNorm(64),
            ReLU(),
            Conv2D(64, 128, (1, 1)),
            BatchNorm(128),
            ReLU(),
            Conv2D(128, 1024, (1, 1)),
            BatchNorm(1024),
            ReLU(),
            MaxPool2D((num_point, 1))
        )
        self.input_fc = Sequential(
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 9,
                   weight_attr=paddle.framework.ParamAttr(
                       initializer=paddle.nn.initializer.Assign(paddle.zeros((256, 9)))),
                   bias_attr=paddle.framework.ParamAttr(
                       initializer=paddle.nn.initializer.Assign(paddle.reshape(paddle.eye(3), [-1])))
                   )
        )
        self.mlp_1 = Sequential(
            Conv2D(1, 64, (1, 3)),
            BatchNorm(64),
            ReLU(),
            Conv2D(64, 64, (1, 1)),
            BatchNorm(64),
            ReLU(),
        )
        self.feature_transform_net = Sequential(
            Conv2D(64, 64, (1, 1)),
            BatchNorm(64),
            ReLU(),
            Conv2D(64, 128, (1, 1)),
            BatchNorm(128),
            ReLU(),
            Conv2D(128, 1024, (1, 1)),
            BatchNorm(1024),
            ReLU(),

            MaxPool2D((num_point, 1))
        )
        self.feature_fc = Sequential(
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 64 * 64)
        )
        self.mlp_2 = Sequential(
            Conv2D(64, 64, (1, 1)),
            BatchNorm(64),
            ReLU(),
            Conv2D(64, 128, (1, 1)),
            BatchNorm(128),
            ReLU(),
            Conv2D(128, 1024, (1, 1)),
            BatchNorm(1024),
            ReLU(),
        )
        self.seg_net = Sequential(
            Conv2D(1088, 512, (1, 1)),
            BatchNorm(512),
            ReLU(),
            Conv2D(512, 256, (1, 1)),
            BatchNorm(256),
            ReLU(),
            Conv2D(256, 128, (1, 1)),
            BatchNorm(128),
            ReLU(),
            Conv2D(128, 128, (1, 1)),
            BatchNorm(128),
            ReLU(),
            Conv2D(128, num_classes, (1, 1)),
            Softmax(axis=1)
        )

    def forward(self, inputs):
        batchsize = inputs.shape[0]

        t_net = self.input_transform_net(inputs)
        t_net = paddle.squeeze(t_net)
        t_net = self.input_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 3, 3])

        x = paddle.reshape(inputs, shape=(batchsize, 1024, 3))
        x = paddle.matmul(x, t_net)
        x = paddle.unsqueeze(x, axis=1)
        x = self.mlp_1(x)

        t_net = self.feature_transform_net(x)
        t_net = paddle.squeeze(t_net)
        t_net = self.feature_fc(t_net)
        t_net = paddle.reshape(t_net, [batchsize, 64, 64])

        x = paddle.reshape(x, shape=(batchsize, 64, 1024))
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.matmul(x, t_net)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.unsqueeze(x, axis=-1)
        point_feat = x
        x = self.mlp_2(x)
        x = paddle.max(x, axis=2)

        global_feat_expand = paddle.tile(paddle.unsqueeze(x, axis=1), [1, self.num_point, 1, 1])
        x = paddle.concat([point_feat, global_feat_expand], axis=1)
        x = self.seg_net(x)
        x = paddle.squeeze(x, axis=-1)
        x = paddle.transpose(x, (0, 2, 1))

        return x