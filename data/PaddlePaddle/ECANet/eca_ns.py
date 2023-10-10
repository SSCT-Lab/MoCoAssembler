import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class eca_layer(paddle.nn.Layer):
    """
    构建ECA模块
    参数：
        channel ：输入特征图的通道数
        k_size ：自适应卷积核的大小
    """
    def __init__(self,channel=1,k_size=3):
        super(eca_layer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.k_size = k_size
        # 如果组数(groups)大于 1，channel等于输入图像通道数除以组数的结果
        self.conv = nn.Conv1D(channel,channel,kernel_size=k_size,bias_attr=False,groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # batch_size ,channel,height,weight 
        b, c, h, w = x.size()

        # 全局信息提取
        y = self.avg_pool(x)

        #对于每一个卷积核覆盖下的区域，元素会被重新排成一列。[N, C, H, W]->[N, Cout, Lout]
        y = nn.functional.unfold(y.transpose(-1,-3),kernel_size=(1,self.k_size),
                            padding=(0,(self.k_size-1)//2 ))
                            
        # ECA模块的两个不同分支
        y = self.conv(y.transpose(-1,-2)).unsqueeze(-1)

        # 多尺度特征融合     
        y = self.sigmoid(y)

        # y变成x一样的形状
        y = y.expand_as(x)

        #y相当于权重，也就是注意力
        result = x*y

        return result
