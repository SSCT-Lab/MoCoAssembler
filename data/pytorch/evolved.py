import torch
import sys
sys.path.append('..')


class AuxiliaryHeadImageNet(torch.nn.Module):
    def __init__(self, channels, num_classes):
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            torch.nn.Conv2d(channels, 128, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 768, 2, bias=False),
            torch.nn.BatchNorm2d(768),
            torch.nn.ReLU(inplace=True)
        )
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
