import jittor
import jittor as jt
import jittor.nn as nn


class DenseNet(nn.Module):
    def __init__(self, density) -> None:
        super().__init__()

        self.density = density

        self.relu = jittor.nn.ReLU()
        self.dropout = jittor.nn.Dropout(0.5)

        self.bn32_1 = jittor.nn.BatchNorm2d(32)
        self.bn32_2 = jittor.nn.BatchNorm2d(32)
        self.bn32_3 = jittor.nn.BatchNorm2d(32)
        self.bn64_1 = jittor.nn.BatchNorm2d(64)
        self.bn64_2 = jittor.nn.BatchNorm2d(64)
        self.bn128_1 = jittor.nn.BatchNorm2d(128)
        self.bn128_2 = jittor.nn.BatchNorm2d(128)
        self.bn128_3 = jittor.nn.BatchNorm2d(128)
        self.bn128_4 = jittor.nn.BatchNorm2d(128)
        self.bn256 = jittor.nn.BatchNorm2d(256)
        self.bn512 = jittor.nn.BatchNorm2d(512)

        self.conv1 = jittor.nn.Conv2d(3, 64, kernel_size=7, stride=2)

        self.pool1 = jittor.nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2_1 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_2 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_3 = jittor.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv64to128 = jittor.nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.conv32to128 = jittor.nn.Conv2d(32, 128, kernel_size=1, stride=1)
        self.conv128to32_1 = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv128to32_2 = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv128to32_3 = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv128to32_4 = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv64to32 = jittor.nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=1)
        self.conv32to256 = jittor.nn.Conv2d(32, 256, kernel_size=1, stride=1)
        self.conv32to512 = jittor.nn.Conv2d(32, 512, kernel_size=1, stride=1)
        self.conv256to128 = jittor.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1)
        self.conv512to128 = jittor.nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=1)

        self.avgpool = jittor.nn.AdaptiveAvgPool2d(1)
        self.softmax = jittor.nn.Softmax(dim=1)

    def execute(self, x):
        features_list = []

        # 1st block
        x = self.conv1(x)
        x = self.bn64_1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 2nd block
        x = self.bn64_2(x)
        x = self.relu(x)
        x = self.conv64to128(x)
        x = self.bn128_1(x)
        x = self.relu(x)
        x = self.conv128to32_1(x)
        x = self.dropout(x)
        features_list.append(x)
        x = jt.concat(features_list, dim=-1)
        for i in range(1, self.density[0]):
            x = x.reshape(-1, 32, 224, 224)
            x = jittor.nn.BatchNorm2d(32)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(32, 128, kernel_size=1, stride=1)(x)
            x = jittor.nn.BatchNorm2d(128)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)(x)
            x = jittor.nn.Dropout(0.5)(x)
            features_list.append(x)
            x = jt.concat(features_list, dim=-1)
        features_list.clear()
        x = self.bn32_1(x)
        x = self.relu(x)
        x = self.conv32to128(x)
        x = self.pool2_1(x)

        # 3rd block
        x = self.bn128_2(x)
        x = self.relu(x)
        x = self.conv128to32_2(x)
        x = self.dropout(x)
        features_list.append(x)
        x = jt.concat(features_list, dim=-1)
        # x = x.reshape(-1, 32, 224, 224)
        for i in range(1, self.density[1]):

            x = jittor.nn.BatchNorm2d(32)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(32, 128, kernel_size=1, stride=1)(x)
            x = jittor.nn.BatchNorm2d(128)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)(x)
            x = jittor.nn.Dropout(0.5)(x)
            features_list.append(x)
            x = jt.concat(features_list, dim=-1)
        features_list.clear()
        x = self.bn32_2(x)
        x = self.relu(x)
        x = self.conv32to256(x)
        x = self.pool2_2(x)

        # 4th block
        x = self.bn256(x)
        x = self.relu(x)
        x = self.conv256to128(x)
        x = self.bn128_3(x)
        x = self.relu(x)
        x = self.conv128to32_3(x)
        x = self.dropout(x)
        features_list.append(x)
        x = jt.concat(features_list, dim=-1)
        # x = x.reshape(-1, 32, 224, 224)
        for i in range(1, self.density[2]):
            x = jittor.nn.BatchNorm2d(32)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(32, 128, kernel_size=1, stride=1)(x)
            x = jittor.nn.BatchNorm2d(128)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)(x)
            x = jittor.nn.Dropout(0.5)(x)
            features_list.append(x)
            x = jt.concat(features_list, dim=-1)
        features_list.clear()
        x = self.bn32_3(x)
        x = self.relu(x)
        x = self.conv32to512(x)
        x = self.pool2_3(x)

        # 5th block
        x = self.bn512(x)
        x = self.relu(x)
        x = self.conv512to128(x)
        x = self.bn128_4(x)
        x = self.relu(x)
        x = self.conv128to32_4(x)
        x = self.dropout(x)
        features_list.append(x)
        x = jt.concat(features_list, dim=-1)
        # x = x.reshape(-1, 32, 224, 224)
        for i in range(1, self.density[3]):
            x = jittor.nn.BatchNorm2d(32)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(32, 128, kernel_size=1, stride=1)(x)
            x = jittor.nn.BatchNorm2d(128)(x)
            x = jittor.nn.ReLU()(x)
            x = jittor.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)(x)
            x = jittor.nn.Dropout(0.5)(x)
            features_list.append(x)
            x = jt.concat(features_list, dim=-1)
        features_list.clear()

        x = self.avgpool(x)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    # model121 = DenseNet([6, 12, 24, 16])
    model = DenseNet([1, 2, 4, 3])
    x = jittor.randn(3, 3, 224, 224)
    y = model(x)
    print(y)
