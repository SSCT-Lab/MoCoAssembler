from paddle import nn


class AlexNet(nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 48, kernel_size=11, stride=4, padding=11 // 2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(48, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(4608, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


def alexnet():
    return AlexNet()
