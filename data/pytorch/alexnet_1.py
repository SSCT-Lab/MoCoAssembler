import torch


class AlexNetOWT_BN(torch.nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(192),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(384),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 6 * 6, 4096, bias=False),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096, bias=False),
            torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_1():
    return AlexNetOWT_BN()
