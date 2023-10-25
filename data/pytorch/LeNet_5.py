from __future__ import print_function
import torch


class BinConv2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
                 Linear=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv

        if dropout != 0:
            self.dropout = torch.nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = torch.nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = torch.nn.Conv2d(input_channels, output_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            if self.previous_conv:
                self.bn = torch.nn.BatchNorm2d(int(input_channels / size), eps=1e-4, momentum=0.1, affine=True)
            else:
                self.bn = torch.nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = torch.nn.Linear(input_channels, output_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        x = self.relu(x)
        return x


class LeNet_5(torch.nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.bn_conv1 = torch.nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
        self.relu_conv1 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_conv2 = BinConv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.bin_ip1 = BinConv2d(50 * 4 * 4, 500, Linear=True,
                                 previous_conv=True, size=4 * 4)
        self.ip2 = torch.nn.Linear(500, 10)

        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.bin_conv2(x)
        x = self.pool2(x)

        x = self.bin_ip1(x)
        x = self.ip2(x)
        return x
