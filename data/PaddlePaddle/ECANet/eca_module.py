import paddle
import paddle.nn as nn
import math
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
        self.conv = nn.Conv1D(1,1,kernel_size=k_size,padding=(k_size-1)//2,bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def update_conv_ksize(self,c,gamma=2,b=1):
        t = int(abs((math.log2(c) + b) /gamma ))
        k_size = t if t%2 else t+1
        self.conv = nn.Conv1D(1,1,kernel_size=k_size, padding=(k_size-1)//2,bias_attr=False)

    def forward(self,x):

        N,C,H,W = x.shape
   
        self.update_conv_ksize(C,2,1)

        # 全局空间信息的特征提取
        y = self.avg_pool(x)

        # ECA模块的两个不同分支 
        # shape = [64,64,1,1]
        y = y.squeeze(-1).transpose([0,2,1])
        # print("infront: ",y.shape)
        y = self.conv(y)
        # print("conv out : ",y.shape)
        y = y.transpose([0,2,1]).unsqueeze(-1)
        # print("conv finish: ",y.shape)

        # 多尺度特征融合     
        y = self.sigmoid(y)

        # y变成x一样的形状
        y = y.expand_as(x)

        #y相当于权重，也就是注意力
        result = x*y

        return result
