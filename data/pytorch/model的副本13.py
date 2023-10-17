import torch


class _Layer_Depwise_Encode(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reserve=False):
        self.stride = int(out_channels / in_channels)
        if reserve:
            self.stride = 1
        super(_Layer_Depwise_Encode, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1,
                            padding=1,
                            groups=in_channels),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class _Layer_Depwise_Decode(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(_Layer_Depwise_Decode, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride,
                            padding=1, groups=in_channel),
            torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class MobileHairNet(torch.nn.Module):
    def __init__(self, nf=32, kernel_size=3, initialize=True):
        super(MobileHairNet, self).__init__()
        self.nf = nf
        self.encode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, stride=2, padding=1),
            _Layer_Depwise_Encode(32, 64, reserve=True)
        )
        self.encode_layer2 = torch.nn.Sequential(
            _Layer_Depwise_Encode(64, 128),
            _Layer_Depwise_Encode(128, 128),
        )
        self.encode_layer3 = torch.nn.Sequential(
            _Layer_Depwise_Encode(128, 256),
            _Layer_Depwise_Encode(256, 256)
        )
        self.encode_layer4 = torch.nn.Sequential(
            _Layer_Depwise_Encode(256, 512),
            _Layer_Depwise_Encode(512, 512),
            _Layer_Depwise_Encode(512, 512),
            _Layer_Depwise_Encode(512, 512),
            _Layer_Depwise_Encode(512, 512),
            _Layer_Depwise_Encode(512, 512),
        )
        self.encode_layer5 = torch.nn.Sequential(
            _Layer_Depwise_Encode(512, 1024),
            _Layer_Depwise_Encode(1024, 1024)
        )
        self.decode_layer1 = torch.nn.Upsample(scale_factor=2)
        self.decode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1),
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            torch.nn.Upsample(scale_factor=2)
        )
        self.decode_layer3 = torch.nn.Sequential(
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            torch.nn.Upsample(scale_factor=2)
        )
        self.decode_layer4 = torch.nn.Sequential(
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            torch.nn.Upsample(scale_factor=2)
        )
        self.decode_layer5 = torch.nn.Sequential(
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            torch.nn.Upsample(scale_factor=2),
            _Layer_Depwise_Decode(in_channel=64, out_channel=64, kernel_size=kernel_size),
            torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=kernel_size, padding=1)
        )
        self.encode_to_decoder4 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.encode_to_decoder3 = torch.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.encode_to_decoder2 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.encode_to_decoder1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.soft_max = torch.nn.Softmax(dim=1)

        if initialize:
            self._init_weight()

    def forward(self, x):
        encode_layer1 = self.encode_layer1(x)
        encode_layer2 = self.encode_layer2(encode_layer1)
        encode_layer3 = self.encode_layer3(encode_layer2)
        encode_layer4 = self.encode_layer4(encode_layer3)
        encode_layer5 = self.encode_layer5(encode_layer4)

        encode_layer4 = self.encode_to_decoder4(encode_layer4)
        encode_layer3 = self.encode_to_decoder3(encode_layer3)
        encode_layer2 = self.encode_to_decoder2(encode_layer2)
        encode_layer1 = self.encode_to_decoder1(encode_layer1)

        decode_layer1 = self.decode_layer1(encode_layer5) + encode_layer4
        decode_layer2 = self.decode_layer2(decode_layer1) + encode_layer3
        decode_layer3 = self.decode_layer3(decode_layer2) + encode_layer2
        decode_layer4 = self.decode_layer4(decode_layer3) + encode_layer1
        decode_layer5 = self.decode_layer5(decode_layer4)

        out = self.soft_max(decode_layer5)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)
