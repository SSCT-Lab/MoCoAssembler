import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear


# VGG模块
class VGGBlock(fluid.dygraph.Layer):
    def __init__(self, in_dim, out_dim, item_num):
        """
        功能:
            初始化VGG模块，H/W=(H/W-F+2*P)/S+1
        输入:
            in_dim   - 输入维度
            out_dim  - 输出维度
            item_num - 项目数量
        输出:
        """
        super(VGGBlock, self).__init__()

        # 添加模块列表
        self.block_list = []  # 模块列表
        for i in range(item_num):
            block_item = self.add_sublayer(  # 构造模块项目
                'block_' + str(i),
                Conv2D(num_channels=in_dim, num_filters=out_dim, filter_size=3, stride=1, padding=1, act='relu'))
            self.block_list.append(block_item)  # 添加模块项目
            in_dim = out_dim  # 设置输入维度

        # 添加最大池化
        self.pool = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

    def forward(self, x):
        """
        功能:
            对输入的特征图进行卷积和池化
        输入:
            x - 输入特征图
        输出:
            x - 输出特征图
        """
        # 进行卷积
        for block_item in self.block_list:
            x = block_item(x)

        # 进行池化
        x = self.pool(x)

        return x


# VGG网络
class VGGNet(fluid.dygraph.Layer):
    def __init__(self):
        """
        功能:
            初始化VGG网络，H/W=(H/W-F+2*P)/S+1
        输入:
        输出:
        """
        super(VGGNet, self).__init__()

        # 添加模组列表
        group_arch = [(3, 64, 2), (64, 128, 2), (128, 256, 3), (256, 512, 3), (512, 512, 3)]  # 模块输入维度，输出维度，项目数量
        self.group_list = []  # 模组列表
        for i, block_arch in enumerate(group_arch):
            group_item = self.add_sublayer(  # 构造模组项目
                'group_' + str(i),
                VGGBlock(in_dim=block_arch[0], out_dim=block_arch[1], item_num=block_arch[2]))
            self.group_list.append(group_item)  # 添加模组项目，输入：N*C*H*W=N*3*32*32，输出：N*C*H*W=N*512*1*1

        # 添加全连接层
        self.fc1 = Linear(input_dim=512, output_dim=512, act='relu')  # 输出：N*C=N*512
        self.drop1 = 0.5  # 训练时随机丢失率
        self.fc2 = Linear(input_dim=512, output_dim=512, act='relu')  # 输出：N*C=N*512
        self.drop2 = 0.5  # 随机丢失
        self.fc3 = Linear(input_dim=512, output_dim=10, act='softmax')  # 输出：N*C=N*2

    def forward(self, x):
        """
        功能:
            对输入图像进行分类
        输入:
            x - 输入图像
        输出:
            x - 预测结果
        """
        # 进行卷积
        for group_item in self.group_list:
            x = group_item(x)

        # 进行预测
        x = fluid.layers.reshape(x=x, shape=[x.shape[0], -1])
        x = self.fc1(x)
        x = fluid.layers.dropout(x=x, dropout_prob=self.drop1)
        x = self.fc2(x)
        x = fluid.layers.dropout(x=x, dropout_prob=self.drop2)
        x = self.fc3(x)

        return x