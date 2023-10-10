import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet50, resnet101


class PSPModule(nn.Layer):
    def __init__(self, num_channels, bin_size_list):
        super(PSPModule, self).__init__()
        self.bin_size_list = bin_size_list
        num_filters = num_channels // len(bin_size_list)
        self.features = []
        for i in range(len(bin_size_list)):
            self.features.append(nn.Sequential(*[
                nn.Conv2D(num_channels, num_filters, 1),
                nn.BatchNorm2D(num_filters),
                nn.ReLU()
            ]))

    def forward(self, inputs):
        out = [inputs]
        for idx, f in enumerate(self.features):
            pool = paddle.nn.AdaptiveAvgPool2D(self.bin_size_list[idx])
            x = pool(inputs)
            x = f(x)
            x = F.interpolate(x, paddle.shape(inputs)[2:], mode="bilinear", align_corners=True)
            out.append(x)

        out = paddle.concat(out, axis=1)
        return out


class PSPNet(nn.Layer):
    def __init__(self, num_classes=13, backbone_name="resnet50"):
        super(PSPNet, self).__init__()

        if backbone_name == "resnet50":
            backbone = resnet50(pretrained=True)
        if backbone_name == "resnet101":
            backbone = resnet101(pretrained=True)

        # self.layer0 = nn.Sequential(*[backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool])
        self.layer0 = nn.Sequential(*[backbone.conv1, backbone.maxpool])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        num_channels = 2048

        self.pspmodule = PSPModule(num_channels, [1, 2, 3, 6])

        num_channels *= 2

        self.classifier = nn.Sequential(*[
            nn.Conv2D(num_channels, 512, 3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Dropout2D(0.1),
            nn.Conv2D(512, num_classes, 1)
        ])

    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pspmodule(x)
        x = self.classifier(x)
        out = F.interpolate(
            x,
            paddle.shape(inputs)[2:],
            mode='bilinear',
            align_corners=True)

        return out
