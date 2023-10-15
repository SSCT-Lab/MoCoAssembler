import torch

__all__ = ['vgg']


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(torch.nn.Module):
    def __init__(self, vgg_name, num_classes=10, batch_norm=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], batch_norm)
        self.classifier = self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 60, 'lr': 1e-1 * (0.2**1)},
            {'epoch': 120, 'lr': 1e-1 * (0.2**2)},
            {'epoch': 160, 'lr': 1e-1 * (0.2**3)}
        ]

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, torch.nn.BatchNorm2d(x), torch.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                in_channels = x
        layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        return torch.nn.Sequential(*layers)
