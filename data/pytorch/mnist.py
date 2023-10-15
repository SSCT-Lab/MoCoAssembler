import torch

__all__ = ['mnist']


class mnist_model(torch.nn.Module):

    def __init__(self):
        super(mnist_model, self).__init__()
        self.feats = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(128)
        )

        self.classifier = torch.nn.Conv2d(128, 10, 1)
        self.avgpool = torch.nn.AvgPool2d(6, 6)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 10)
        return out


def mnist(**kwargs):
    return mnist_model()
