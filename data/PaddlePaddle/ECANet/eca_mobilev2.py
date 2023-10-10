import paddle.nn as nn
from models.eca_module import eca_layer 
import numpy as np
import paddle
from paddle.utils.download import get_weights_path_from_url

__all__ = ['ECA_MobileNetV2', 'eca_mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2D(in_planes, out_planes, kernel_size, stride, padding, groups=groups,bias_attr=False),     
            nn.BatchNorm2D(out_planes),
            nn.ReLU6(True)
        )

class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, k_size):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(oup,momentum=0.1),
        ])
        layers.append(eca_layer(oup, k_size))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)        



class ECA_MobileNetV2(nn.Layer):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(ECA_MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if c < 96:
                    ksize = 1
                else:
                    ksize = 3
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, k_size=ksize))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal)
                if m.bias is not None:
                    m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
            elif isinstance(m, nn.Linear):
                v = np.random.normal(loc=0,scale=0.01,size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
                if m.bias is not None:
                   m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def forward(self, x):
        x = self.features(x)
        x = x.mean(-1).mean(-1)
        x = self.classifier(x)
        return x



def eca_mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ECA_MobileNetV2 architecture from
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ECA_MobileNetV2(**kwargs)
    if pretrained:
        weight_path = get_weights_path_from_url(model_urls["mobilenet_v2"][0], model_urls["mobilenet_v2"][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


    