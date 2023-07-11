import torch
import torch.nn as nn

class MobileNet(nn.Module):
    def __init__(self, in_channels):
        super(MobileNet, self).__init__()

        self.fc = torch.nn.Linear(in_features=1024, out_features=1000)
        self.relu1a = torch.nn.ReLU()
        self.relu1b = torch.nn.ReLU()
        self.relu2a = torch.nn.ReLU()
        self.relu2b = torch.nn.ReLU()
        self.relu3a = torch.nn.ReLU()
        self.relu3b = torch.nn.ReLU()
        self.relu4a = torch.nn.ReLU()
        self.relu4b = torch.nn.ReLU()
        self.relu5a = torch.nn.ReLU()
        self.relu5b = torch.nn.ReLU()
        self.relu6a = torch.nn.ReLU()
        self.relu6b = torch.nn.ReLU()
        self.relu7a = torch.nn.ReLU()
        self.relu7b = torch.nn.ReLU()
        self.relu8a = torch.nn.ReLU()
        self.relu8b = torch.nn.ReLU()
        self.relu9a = torch.nn.ReLU()
        self.relu9b = torch.nn.ReLU()
        self.relu10a = torch.nn.ReLU()
        self.relu10b = torch.nn.ReLU()
        self.relu11a = torch.nn.ReLU()
        self.relu11b = torch.nn.ReLU()
        self.relu12a = torch.nn.ReLU()
        self.relu12b = torch.nn.ReLU()
        self.relu13a = torch.nn.ReLU()
        self.relu13b = torch.nn.ReLU()

        self.conv0 = torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)

        self.conv1a = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1b = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv2a = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64)
        self.conv2b = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv3a = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)
        self.conv3b = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv4a = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128)
        self.conv4b = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv5a = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256)
        self.conv5b = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv6a = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, groups=256)
        self.conv6b = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv7a = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv7b = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv8a = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv8b = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv9a = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv9b = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv10a = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv10b = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv11a = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512)
        self.conv11b = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv12a = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, groups=512)
        self.conv12b = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.conv13a = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=4, groups=1024)
        self.conv13b = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        # 1st block
        x = self.conv0(x)

        # 2nd block
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.relu1b(x)

        # 3rd block
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.relu2b(x)

        # 4th block
        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.relu3b(x)

        # 5th block
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.relu4b(x)

        # 6th block
        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.relu5b(x)

        # 7th block
        x = self.conv6a(x)
        x = self.relu6a(x)
        x = self.conv6b(x)
        x = self.relu6b(x)

        # 8th block
        x = self.conv7a(x)
        x = self.relu7a(x)
        x = self.conv7b(x)
        x = self.relu7b(x)

        # 9th block
        x = self.conv8a(x)
        x = self.relu8a(x)
        x = self.conv8b(x)
        x = self.relu8b(x)

        # 10th block
        x = self.conv9a(x)
        x = self.relu9a(x)
        x = self.conv9b(x)
        x = self.relu9b(x)

        # 11th block
        x = self.conv10a(x)
        x = self.relu10a(x)
        x = self.conv10b(x)
        x = self.relu10b(x)

        # 12th block
        x = self.conv11a(x)
        x = self.relu11a(x)
        x = self.conv11b(x)
        x = self.relu11b(x)

        # 13th block
        x = self.conv12a(x)
        x = self.relu12a(x)
        x = self.conv12b(x)
        x = self.relu12b(x)

        # 14th block
        x = self.conv13a(x)
        x = self.relu13a(x)
        x = self.conv13b(x)
        x = self.relu13b(x)

        # 15th block
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        # output
        x = self.fc(x)
        return x

def go():
    device = torch.device('cuda')
    model = MobileNet(3).to(device)
    x = torch.randn((4, 3, 224, 224)).to(device)
    y = model(x)
    return model
