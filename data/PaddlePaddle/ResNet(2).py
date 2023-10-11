import math
from functools import partial

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from paddle import fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv3D, BatchNorm, Linear, to_variable

def get_inplanes():
    return [64, 128, 256, 512]
class ReLU(fluid.dygraph.Layer):
    def forward(self, x):
        return fluid.layers.relu(x)


class AvgPool3d(fluid.dygraph.Layer):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return fluid.layers.pool3d(x,
                                   pool_size=self.kernel_size,
                                   pool_stride=self.stride,
                                   pool_padding=self.padding,
                                   pool_type='avg')

class AdaptiveAvgPool3d(fluid.dygraph.Layer):
    def __init__(self, output_size):
        super(AdaptiveAvgPool3d, self).__init__()
        self.output_size = output_size
    def forward(self, x):
        return fluid.layers.adaptive_pool3d(x,
                                            pool_size=self.output_size,
                                            pool_type='avg')


class MaxPool3d(fluid.dygraph.Layer):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return fluid.layers.pool3d(x,
                                   pool_size=self.kernel_size,
                                   pool_stride=self.stride,
                                   pool_padding=self.padding)

def conv3x3x3(in_planes, out_planes, stride=1):
    return Conv3D(in_planes,
                     out_planes,
                     filter_size=3,
                     stride=stride,
                     padding=1,
                     bias_attr=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return Conv3D(in_planes,
                     out_planes,
                     filter_size=1,
                     stride=stride,
                     bias_attr=False)


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = BatchNorm(planes, act='relu')
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = BatchNorm(planes, act='relu')
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = BatchNorm(planes, act='relu')
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(out)



class ResNet(fluid.dygraph.Layer):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = Conv3D(n_input_channels,
                               self.in_planes,
                               filter_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias_attr=False)
        self.bn1 = BatchNorm(self.in_planes, act='relu')
        self.maxpool = MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = AdaptiveAvgPool3d((1, 1, 1))
        self.fc_in_dim = block_inplanes[3] * block.expansion
        self.fc = Linear(self.fc_in_dim, n_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight,
        #                                 mode='fan_out',
        #                                 nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = fluid.layers.pool3d(x, pool_size=1, pool_stride=stride, pool_type='avg')
        # zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
        #                         out.size(3), out.size(4))
        shape = out.shape
        zero_pads = fluid.layers.zeros([shape[0], planes - shape[1], shape[2], shape[3], shape[4]],
                                       dtype='float32')
        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()

        # out = torch.cat([out.data, zero_pads], dim=1)
        out = fluid.layers.concat([out, zero_pads], axis=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = fluid.dygraph.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    BatchNorm(planes * block.expansion)
                )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # x = x.view(x.size(0), -1)
        x = fluid.layers.reshape(x, [x.shape[0],-1])
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

import numpy as np

def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')

if __name__ == '__main__':
    use_gpu = False
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):



        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_classes=1039)
        model.train()
        state_dic,_ = fluid.dygraph.load_dygraph('../paddle_resnet50_mk.pdparams')
        model.set_dict(state_dic)
        count = 0
        with open('train_params.txt', 'w') as f:
            for p in model.named_parameters():
                f.write(p[0] + '\n')

        params_list = []
        for k,v in model.named_parameters():
            if '_mean' not in k and '_variance' not in k:
                params_list.append(v)

        opt = fluid.optimizer.Momentum(
            learning_rate=3e-3,
            momentum=0.9,
            parameter_list=model.parameters(),
            # parameter_list=parameters,
            regularization=fluid.regularizer.L2Decay(0.2)
        )
        nllloss = fluid.dygraph.NLLLoss()
        # data = np.random.uniform(0,1, size=(1,3,32,224,224)).astype('float32')
        # np.save('data.npy',data)
        for i in range(20):

            data = np.load('data.npy')
            data = data[:,:,:1,:,:]
            data = to_variable(data)
            label = to_variable(np.array([[1]]))
            label.stop_gradient = True
            data.stop_gradient = True
            y = model(data)
            # y = fluid.layers.softmax(y)
            # y = fluid.layers.log(y)
            # loss = nllloss(y, label)
            # loss = fluid.layers.cross_entropy(y, label)
            # loss = fluid.layers.mean(loss)
            loss = fluid.layers.reduce_mean(y)
            loss = fluid.layers.square(loss-1)
            print(f'epoch:{i} loss:{loss.numpy()}')
            np.save(f'loss_{i}.npy', loss.numpy()[0])
            # print(f'epoch:{i} y:{y_mean.numpy()}')
            loss.backward()
            opt.minimize(loss)
            model.clear_gradients()
        pass