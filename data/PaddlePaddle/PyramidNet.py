import paddle
import paddle.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)        
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.shape[2:4]
        else:
            shortcut = x
            featuremap_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]

        if residual_channel != shortcut_channel:
            padding = paddle.zeros([batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]]) 
            out += paddle.concat([shortcut, padding], 1)
        else:
            out += shortcut 

        return out


class Bottleneck(nn.Layer):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, (planes*1), kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D((planes*1))
        self.conv3 = nn.Conv2D((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias_attr=False)
        self.bn4 = nn.BatchNorm2D(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.shape[2:4]
        else:
            shortcut = x
            featuremap_size = out.shape[2:4]

        batch_size = out.shape[0]
        residual_channel = out.shape[1]
        shortcut_channel = shortcut.shape[1]

        if residual_channel != shortcut_channel:
            padding = paddle.zeros([batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]]) 
            out += paddle.concat([shortcut, padding], 1)
        else:
            out += shortcut 

        return out


class PyramidNet(nn.Layer):
        
    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=False):
        super(PyramidNet, self).__init__()   	
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.addrate = alpha / (3*n*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2D(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias_attr=False)
            self.bn1 = nn.BatchNorm2D(self.input_featuremap_dim)

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2D(self.final_featuremap_dim)
            self.relu_final = nn.ReLU()
            self.avgpool = nn.AvgPool2D(8)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        elif dataset == 'imagenet':
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

            if layers.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth-2)/12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth-2)/8)

                layers[depth]= [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers[depth])

            self.inplanes = 64        
            self.addrate = alpha / (sum(layers[depth])*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2D(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias_attr=False)
            self.bn1 = nn.BatchNorm2D(self.input_featuremap_dim)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= nn.BatchNorm2D(self.final_featuremap_dim)
            self.relu_final = nn.ReLU()
            self.avgpool = nn.AvgPool2D(7) 
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight.set_value(paddle.normal(mean=0, std=math.sqrt(2. / n), shape=m.weight.shape))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2D((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = paddle.reshape(x, [x.shape[0], -1])
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = paddle.reshape(x, [x.shape[0], -1])
            x = self.fc(x)
    
        return x

if __name__ == '__main__':
    network = PyramidNet('imagenet', 32, 300, num_classes=10, bottleneck=True)
    img = paddle.zeros([1, 3, 224, 224])
    outs = network(img)
    print(outs.shape)