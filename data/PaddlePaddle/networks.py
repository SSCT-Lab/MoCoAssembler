import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D,  Linear

class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        print('input_nc=',input_nc)
        self.output_nc = output_nc
        print('output_nc=',output_nc)
        self.ngf = ngf
        print('ngf=',ngf)
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [#ReflectionPad2d(3),0
                      Conv2D(input_nc,ngf,filter_size=7,stride=1,padding=3,bias_attr=False),
                    #   paddle.fluid.layers.instance_norm(ngf),
                      InstanceNorm2d(ngf),
                    #   paddle.nn.ReLU(inplace=True)
                      ReLU(inplace=True)
                      ]
        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [#ReflectionPad2d(1),0
                         Conv2D(ngf * mult,ngf * mult * 2,filter_size=3,stride=2,padding=1,bias_attr=False),
                         InstanceNorm2d(ngf * mult * 2),
                         ReLU(inplace=True)
                         ]
        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(input_dim=ngf * mult,output_dim=1,  bias_attr=False)
        self.gmp_fc =Linear(input_dim=ngf * mult,output_dim=1,  bias_attr=False)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU(inplace=True)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(input_dim=ngf * mult,output_dim=ngf * mult,  bias_attr=False),
                  ReLU(inplace=True),
                  Linear(input_dim=ngf * mult,output_dim=ngf * mult,  bias_attr=False),
                  ReLU(inplace=True)]
        else:
            FC = [Linear(input_dim=img_size // mult * img_size // mult * ngf * mult,
                         output_dim=ngf * mult,  bias_attr=False),
                  ReLU(inplace=True),
                  Linear(input_dim=ngf * mult,output_dim=ngf * mult,  bias_attr=False),
                  ReLU(inplace=True)]
        self.gamma =  Linear(input_dim=ngf * mult, output_dim=ngf * mult,  bias_attr=False)
        self.beta =  Linear(input_dim=ngf * mult,output_dim=ngf * mult,  bias_attr=False)
        # paddle.fluid.dygraph.Linear



        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
                          # fluid.layers.resize_nearest(scale=2) 
            UpBlock2 += [
                         Upsample(scale=2),###########
                        #  ReflectionPad2d(1),0
                         Conv2D(ngf * mult, int(ngf * mult / 2),filter_size=3,stride=1,padding=1,bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU(inplace=True)]


        UpBlock2 += [#ReflectionPad2d(3), 0                                  ###tanh被我内嵌了###
                     Conv2D(ngf, output_nc,filter_size=7,stride=1,padding=3,bias_attr=False,act='tanh'),
                    #  paddle.fluid.layers.tanh(x, name=None)
                     ]

        self.DownBlock = paddle.fluid.dygraph.Sequential(*DownBlock)
        self.FC = paddle.fluid.dygraph.Sequential(*FC)
        self.UpBlock2 = paddle.fluid.dygraph.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = paddle.fluid.layers.adaptive_pool2d(x, 1)
        # print('--------------gap',gap)
        # print('-------------x.shape',x.shape[0])
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_logit=self.gap_fc(fluid.layers.reshape(gap,shape=[x.shape[0], -1])) 
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = paddle.fluid.layers.transpose(gap_weight,perm=[1,0])##################
        # print("--gap_weight",gap_weight.shape) 
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        temp = fluid.layers.unsqueeze(gap_weight,2)
        temp2 = fluid.layers.unsqueeze(temp,3)
        # print('gap----------',gap.shape)
        gap = x * temp2              #([1, 256，64，64])
        # print('gap----------',gap.shape)

        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)##########
        # fluid.layers.unsqueeze(input, axes, name=None)
        gmp = paddle.fluid.layers.adaptive_pool2d(x, 1)
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp,shape=[x.shape[0], -1]))
        # print('gmp----------',gmp.shape)
        # print('gmp_logit----------',gmp_logit.shape)
        # print('gap_weight----------',gap_weight.shape)
        
        gmp_weight = list(self.gmp_fc.parameters())[0]
        # print('gmp_weight----------',gmp_weight.shape)

        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        gmp_weight = paddle.fluid.layers.transpose(gmp_weight,perm=[1,0])
        # print('gmp_weight----------',gmp_weight.shape)

        t = fluid.layers.unsqueeze(gmp_weight,2)
        t2 = fluid.layers.unsqueeze(t,3)
        gmp = x * t2
        # print('gmp----------',gmp.shape)

        cam_logit = paddle.fluid.layers.concat([gap_logit, gmp_logit], 1)
        # print('cam_logit----------',cam_logit.shape)
        # print('gap----------',gap.shape)
        # print('gmp----------',gmp.shape)
        x = paddle.fluid.layers.concat([gap, gmp], 1)
        # print('x = paddle.fluid.layers.concat([gap, gmp], 1)----------',x.shape)
        # x = self.relu(self.conv1x1(x))
        x = self.relu(self.conv1x1(x))###################
        # x = paddle.fluid.layers.relu(self.conv1x1(x))#########################
        # heatmap=fluid.layers.sum(x)##############################################
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = paddle.fluid.layers.adaptive_pool2d(x, 1)
            x_ = self.FC(fluid.layers.reshape(x_,shape=[x_.shape[0], -1]))

        else:
            x_ = self.FC(fluid.layers.reshape(x,shape=[x.shape[0], -1]))
        # print('-------------------x_',x_)
        # gamma, beta = self.gamma(x_), self.beta(x_)  
        gamma = self.gamma(x_)
        beta = self.beta(x_) 

        for i in range(self.n_blocks):
            # print('gamma-------',gamma.shape)
            # print('beta----------',beta.shape)
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
            # print('x-getattr---------',x.shape)
        out = self.UpBlock2(x)
        # print('out = self.UpBlock2(x)-ResnetGenerator-forward---------',out.shape)
        # print('cam_logit---------',cam_logit.shape)
        # print('heatmap---------',heatmap.shape)

        return out, cam_logit, heatmap


class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale
    def forward(self, inputs):
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        # self.pad1 =  ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm1 = adaILN(dim)############
        self.relu1 = ReLU(inplace=True)

        # self.pad2 = ReflectionPad2d(1)0
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm2 = adaILN(dim)
##############################################
    def forward(self, x, gamma, beta):
        # out = self.pad1(x)
        out = self.conv1(x)
        # out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        # out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x




class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()

        conv_block = []
        conv_block += [
                    #   ReflectionPad2d(1),
                      Conv2D(dim,dim,filter_size=3,stride=1,padding=1,bias_attr=use_bias),
                      InstanceNorm2d(dim),
                      ReLU(inplace=True)
                      ]

        conv_block += [
                    #   ReflectionPad2d(1),
                      Conv2D(dim,dim,filter_size=3,stride=1,padding=1,bias_attr=use_bias),
                      InstanceNorm2d(dim),
                      ]
        self.conv_block = paddle.fluid.dygraph.Sequential(*conv_block)


    def forward(self,x):
        out = x + self.conv_block(x)
        return out



class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        # shape1 = np.array([1, num_features, 1, 1], dtype=np.int32)########
        #https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/layers_cn/create_parameter_cn.html#cn-api-fluid-layers-create-parameter
        # self.rho = fluid.layers.create_parameter(x)#########
        # self.rho = fluid.layers.create_parameter(shape1,dtype='float32',
        #     default_initializer= fluid.initializer.ConstantInitializer(0.9))#Constant
        self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.9, dtype='float32')

    def forward(self, input, gamma, beta):
        # print("adaIN_input",input.shape) # torch.Size([1, 256, 64, 64])
        in_mean = paddle.fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True, name=None)
        in_var = var(input, dim=[2, 3], keep_dim=True)#########
        # print("in_mean",in_mean.shape) # torch.Size([1, 256, 1, 1])
        # print("in_var", in_var.shape) #  torch.Size([1, 256, 1, 1])
        out_in = (input - in_mean) / paddle.fluid.layers.sqrt(in_var + self.eps)
        ln_mean = paddle.fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True, name=None)
        ln_var =  var(input, dim=[1, 2, 3], keep_dim=True)
        # print("ln_mean", ln_mean.shape) #  torch.Size([1, 1, 1, 1])
        # print("ln_var", ln_var.shape) # torch.Size([1, 1, 1, 1])
        # ln_var = fluid.layers.reduce_mean((input - in_mean) ** 2, dim=[1,2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / paddle.fluid.layers.sqrt(ln_var + self.eps)
        # print("out_ln", out_ln.shape) # torch.Size([1, 256, 64, 64])
        rho_expand = paddle.fluid.layers.expand(self.rho, (input.shape[0], 1, 1, 1))##########
        out = rho_expand * out_in + (1- rho_expand) * out_ln###########
        # out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        temp = fluid.layers.unsqueeze(gamma,2)
        temp2 = fluid.layers.unsqueeze(temp,3)
        c1 = out * temp2
        tem = fluid.layers.unsqueeze(beta,2)
        tem2 = fluid.layers.unsqueeze(tem,3)
        c2 =  tem2
        out = c1 + c2
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)##########
        # print("out--adaILN", out.shape) # torch.Size([1, 256, 64, 64])
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        # self.pad1 =  ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm1 = adaILN(dim)############
        self.relu1 = ReLU(inplace=True)

        # self.pad2 = ReflectionPad2d(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        # print("ResnetAdaILNBlock x",x.shape) # torch.Size([1, 256, 64, 64])
        # out = self.pad1(x)
        # print("out1-pad", out.shape) # torch.Size([1, 256, 66, 66])
        out = self.conv1(x)
        # out = self.conv1(out)
        # print("out2-conv1", out.shape) 
        out = self.norm1(out, gamma, beta)
        # print("out_norm1", out.shape) # torch.Size([1, 256, 64, 64])
        out = self.relu1(out)
        # print("out-relu1", out.shape) # torch.Size([1, 256, 64, 64])
        # out = self.pad2(out)
        # print("out-pad2--", out.shape) # torch.Size([1, 256, 64, 64])  66？？
        out = self.conv2(out)
        # print("out-conv2", out.shape) # torch.Size([1, 256, 64, 64])
        out = self.norm2(out, gamma, beta)
        # print("out_norm2", out.shape) # torch.Size([1, 256, 64, 64])
        return out + x




class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps

        # shape1 = np.array([1, num_features, 1, 1], dtype=np.int32)########

        # self.rho = fluid.layers.create_parameter(shape1,dtype='float32')
        self.rho = fluid.layers.fill_constant([1, num_features, 1, 1], value=0.0, dtype='float32')
        # self.rho.data.fill_(0.0)
        # self.gamma = fluid.layers.create_parameter(shape1,dtype='float32',
        # default_initializer= fluid.initializer.ConstantInitializer(value=1.0))
        self.gamma = fluid.layers.fill_constant([1, num_features, 1, 1], value=1.0, dtype='float32')
        # self.beta = fluid.layers.create_parameter(shape1,dtype='float32')
        self.beta = fluid.layers.fill_constant([1, num_features, 1, 1], value=0.0, dtype='float32')

    def forward(self, input):
        # print("ILN_input-------------",input.shape) # torch.Size([1, 256, 64, 64])
        in_mean = paddle.fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        # print("in_mean",in_mean.shape) # torch.Size([1, 256, 1, 1])
        in_var = var(input, dim=[2, 3], keep_dim=True)#########
        # print("in_var", in_var.shape) #  torch.Size([1, 256, 1, 1])
        out_in = (input - in_mean) / paddle.fluid.layers.sqrt(in_var + self.eps)
        ln_mean = paddle.fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = var(input, dim=[1, 2, 3], keep_dim=True)
        # print("ln_var", ln_var.shape) # torch.Size([1, 1, 1, 1])
        out_ln = (input - ln_mean) / paddle.fluid.layers.sqrt(ln_var + self.eps)
        # print("out_ln", out_ln.shape) # torch.Size([1, 256, 64, 64])
        rho_expand = paddle.fluid.layers.expand(self.rho, (input.shape[0], 1, 1, 1))##########
        out = rho_expand * out_in + (1- rho_expand) * out_ln###########
        gamma_expand = paddle.fluid.layers.expand(self.gamma, (input.shape[0], 1, 1, 1))
        beta_expand = paddle.fluid.layers.expand(self.beta, (input.shape[0], 1, 1, 1))
        out = out * gamma_expand + beta_expand
        # print("ILN---out", out.shape)
        return out

# def Spectralnorm(layer, name='weight', n_power_iterations=1, eps=1e-12, dim=0):
#     # paddle.fluid.dygraph.Layer.register_forward_pre_hook

#     def spectral_norm_pre_hook(layer, x):
#         layer.weight = paddle.fluid.layers.spectral_norm(layer.weight, dim=dim, power_iters=n_power_iterations, eps=eps)
#         # return x

#     layer.register_forward_pre_hook(spectral_norm_pre_hook)

#     return layer


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [#ReflectionPad2d(1),
                 Spectralnorm(
                     Conv2D(input_nc, ndf,filter_size=4,stride=2,padding=1,bias_attr=True,act='leaky_relu'))
        ]
                #  https://github.com/PaddlePaddle/Paddle/issues/26139
                #  nn.LeakyReLU(0.2, True)

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [#ReflectionPad2d(1),
                      Spectralnorm(
                      Conv2D(ndf * mult, ndf * mult * 2,filter_size=4,stride=2,padding=1,bias_attr=True,act='leaky_relu')),
                    #   nn.LeakyReLU(0.2, True)
                      ]
        mult = 2 ** (n_layers - 2 - 1)
        model += [#ReflectionPad2d(1),
                  Spectralnorm(
                      Conv2D(ndf * mult, ndf * mult * 2,filter_size=4,stride=1,padding=1,bias_attr=True,act='leaky_relu')),
                    #   nn.LeakyReLU(0.2, True)
                    ]
        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(paddle.fluid.dygraph.Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(paddle.fluid.dygraph.Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult,filter_size=1,stride=1,bias_attr=True,act='leaky_relu')
        # self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.leaky_relu = leaky_relu(0.2, True)
        # self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(
                      Conv2D(ndf * mult, 1,filter_size=4,stride=1,padding=1,bias_attr=False))
        self.model = paddle.fluid.dygraph.Sequential(*model)  

    def forward(self, input):
        x = self.model(input)#####################

        gap = paddle.fluid.layers.adaptive_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_logit = self.gap_fc(fluid.layers.reshape(gap,shape=[x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = paddle.fluid.layers.transpose(gap_weight,perm=[1,0])##################
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        t = fluid.layers.unsqueeze(gap_weight,2)
        t2 = fluid.layers.unsqueeze(t,3)
        gap = x * t2

        gmp = paddle.fluid.layers.adaptive_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp,shape=[x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = paddle.fluid.layers.transpose(gmp_weight,perm=[1,0])##################
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        tt = fluid.layers.unsqueeze(gmp_weight,2)
        tt2 = fluid.layers.unsqueeze(tt,3)
        gmp = x * tt2

        cam_logit = paddle.fluid.layers.concat([gap_logit, gmp_logit], 1)
        # print('Discriminator gap--------------',gap.shape)
        # print('Discriminator gmp--------------',gmp.shape)
        x = paddle.fluid.layers.concat([gap, gmp], 1)
        # print('x=paddle.fluid.layers.concat([gap, gmp], 1) --------------',x.shape)
        x = self.leaky_relu(self.conv1x1(x))
        # print('self.leaky_relu(self.conv1x1(x)) --------------',x.shape)

        # heatmap = paddle.tensor.sum(x, axis=1, keepdim=True)
        # heatmap = fluid.layers.sum(x)
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)


        # x = self.pad(x)
        out = self.conv(x)
        # print('cam_logit Discriminator--------------',cam_logit.shape)
        # print('heatmap Discriminator--------------',heatmap.shape)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

#https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/nn_cn/pad2d_cn.html#pad2d
class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")


#https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/declarative_cn/instance_norm_cn.html#instance-norm
class InstanceNorm2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(InstanceNorm2d, self).__init__()
        self.size = size

    def forward(self, x):
        return paddle.fluid.layers.instance_norm(x)

class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace
        # ReLU(self)

    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.relu(x))
            return x
        else:
            y = fluid.layers.relu(x)
            return y

class leaky_relu(fluid.dygraph.Layer):
    def __init__(self, alpha=0.02,inplace=False):#alpha (float) - 负斜率，缺省值为0.02。
        super(leaky_relu, self).__init__()
        self.inplace = inplace
        self.alpha = alpha

    def forward(self, x):
        if self.inplace:
            x.set_value(paddle.fluid.layers.leaky_relu(x,self.alpha))
            return x
        else:
            y = paddle.fluid.layers.leaky_relu(x,self.alpha)
            return y

def var(input,dim=None,keep_dim=False,unbiased=True,name=None):
    rank = len(input.shape)
    dims = dim if dim != None and dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in dims]
    inp_shape = input.shape
    mean = fluid.layers.reduce_mean(input,dim=dim,keep_dim=True,name=name)
    # print('mean---------',mean)#[256, 256, 1, 1]
    # print('(input-mean)**2=',(input-mean)**2)
    tmp = fluid.layers.reduce_mean(( input-mean)**2,dim=dim,keep_dim=keep_dim,name=name)
    # tmp = fluid.layers.reduce_mean((mean)**2,dim=dim,keep_dim=keep_dim,name=name)

    if unbiased:
        n=1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    return tmp


import paddle.fluid.dygraph.nn as nn
class Spectralnorm(fluid.dygraph.Layer ):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

