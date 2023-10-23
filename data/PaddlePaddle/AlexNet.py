import paddle


class AlexNet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = paddle.nn.Sequential(
            paddle.nn.Conv2D(3, 48, kernel_size=11, stride=4, padding=11 // 2),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(48, 128, kernel_size=5, padding=2),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(128, 192, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(192, 192, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(192, 128, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.classifier = paddle.nn.Sequential(
            paddle.nn.Linear(3 * 3 * 128, 2048),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(0.5),
            paddle.nn.Linear(2048, 2048),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(0.5),
            paddle.nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)

        return x
