import math
import paddle.nn as nn
from models.eca_module import eca_layer
from paddle.utils.download import get_weights_path_from_url
import numpy as np
import paddle

__all__ = []

model_urls = {
    "resnet18": ("https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams", "cf548f46534aa3560945be4b95cd11c4"),
    "resnet34": ("https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams", "8d2275cf8706028345f78ac0e1d31969"),
    "resnet50": ("https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams", "ca6f485ee1ab0492d38f323885b0ad80"),
    "resnet101": ("https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams", "02f35f034ca3858e1e54d4036443c92d"),
    "resnet152": ("https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams", "7ad16a2f1e7333859ff986138630fd7a"),
    "wide_resnet50_2": ("https://paddle-hapi.bj.bcebos.com/models/wide_resnet50_2.pdparams", "0282f804d73debdab289bd9fea3fa6dc"),
    "wide_resnet101_2": ("https://paddle-hapi.bj.bcebos.com/models/wide_resnet101_2.pdparams",
"d4360a2d23657f059216f5d5a1a9ac93")
}


def conv3x3(in_planes,out_planes,stride=1):
    """
    3x3的卷积核,可设置步幅
    """
    kernel = nn.Conv2D(in_planes,out_planes,kernel_size=3,stride=stride,
            padding=1,bias_attr=False)
    return kernel

class ECABasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = conv3x3(planes,planes,1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.eca = eca_layer(planes,k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual     # 残差跨越
        out = self.relu(out) 

        return out



class ECABottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU(True)
        self.eca = eca_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

        


class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        width (int): base width of resnet, default: 64.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.

    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            resnet50 = ResNet(BottleneckBlock, 50)

            wide_resnet50_2 = ResNet(BottleneckBlock, 50, width=64*2)

            resnet18 = ResNet(BasicBlock, 18)

            x = paddle.rand([1, 3, 224, 224])
            out = resnet18(x)

            print(out.shape)

    """

    def __init__(self, block, depth=50, width=64, num_classes=1000,k_size=[3,3,3,3], with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        layers = layer_cfg[depth]
        self.groups = 1
        self.base_width = width
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias_attr=False)
        
                            
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1],int(k_size[0]) ,stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],int(k_size[0]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],int(k_size[0]), stride=2)
        if with_pool:
            self.avgpool = nn.AvgPool2D(7,stride=1)

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0]*m.weight.shape[1]*m.weight.shape[2]
                v = np.random.normal(loc=0.,scale=np.sqrt(2./n),size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))


    def _make_layer(self, block, planes, blocks, k_size ,stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, k_size
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes,k_size=k_size)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x

def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    if pretrained:
        assert (
            arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0], model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    print("Constructing eca_resnet18......")
    if pretrained :
        model = _resnet("resnet18", ECABasicBlock, 18, pretrained, **kwargs)
        model.avgpool = nn.AdaptiveAvgPool2D(1)
        return model
    else:
        model = ResNet(ECABasicBlock, 18 , num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2D(1)
        return model

def eca_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing eca_resnet50......")
    if pretrained :
        model = _resnet("resnet18", ECABottleneck, 18, pretrained, **kwargs)
        model.avgpool = nn.AdaptiveAvgPool2D(1)
        return model
    else: 
        model = ResNet(ECABottleneck, 50 , num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model        

def get_model(method):
    if method == "eca_resnet18":
        return eca_resnet18
    elif method == "eca_resnet50":
        return eca_resnet50
    else:
        assert "models hasn't this net !"
