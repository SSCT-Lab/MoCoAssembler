import os
import paddle
import paddle.nn as nn

__all__ = [
    "hardnet68", "hardnet85", "hardnet68ds",
    "hardnet39ds"
]
class Flatten(nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return paddle.reshape(x,[x.shape[0],-1])



class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_sublayer('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_sublayer('layer2',DWConvLayer(out_channels, out_channels, stride=stride))
        
    def forward(self, x):
        return super().forward(x)

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels
        
        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')
        
        self.add_sublayer('dwconv', nn.Conv2D(groups, groups, filter_size=3,
                                          stride=stride, padding=1, groups=groups))
        self.add_sublayer('norm', nn.BatchNorm2D(groups))
    def forward(self, x):
        return super().forward(x)  

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_sublayer('conv', nn.Conv2D(in_channels, out_ch, kernel,stride=stride, padding=kernel//2, groups=groups))
        self.add_sublayer('norm', nn.BatchNorm2D(out_ch))
        self.add_sublayer('relu', nn.ReLU6(True))                                          
    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Layer):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch))
          else:
            layers_.append(ConvLayer(inch, outch))
          
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.LayerList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = paddle.concat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = paddle.concat(out_, 1)
        return out
        
        
        
        
class HarDNet(nn.Layer):
    def __init__(self, class_dim=1000,depth_wise=False, arch=85, pretrained=True, weight_path=''):
        super().__init__()
        first_ch  = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1
        
        #HarDNet68
        ch_list = [  128, 256, 320, 640, 1024]
        gr       = [  14, 16, 20, 40,160]
        n_layers = [   8, 16, 16, 16,  4]
        downSamp = [   1,  0,  1,  1,  0]
        
        if arch==85:
          #HarDNet85
          first_ch  = [48, 96]
          ch_list = [  192, 256, 320, 480, 720, 1280]
          gr       = [  24,  24,  28,  36,  48, 256]
          n_layers = [   8,  16,  16,  16,  16,   4]
          downSamp = [   1,   0,   1,   0,   1,   0]
          drop_rate = 0.2
        elif arch==39:
          #HarDNet39
          first_ch  = [24, 48]
          ch_list = [  96, 320, 640, 1024]
          grmul = 1.6
          gr       = [  16,  20, 64, 160]
          n_layers = [   4,  16,  8,   4]
          downSamp = [   1,   1,  1,   0]
          
        if depth_wise:
          second_kernel = 1
          max_pool = False
          drop_rate = 0.05
        
        blks = len(n_layers)
        self.base = nn.LayerList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2) )
  
        # Second Layer
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=second_kernel) )
        
        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
          self.base.append(nn.MaxPool2D(kernel_size=3, stride=2, padding=1))
        else:
          self.base.append ( DWConvLayer(first_ch[1], first_ch[1], stride=2) )

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append ( blk )
            
            if i == blks-1 and arch == 85:
                self.base.append ( nn.Dropout(0.1))
            
            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            if downSamp[i] == 1:
              if max_pool:
                self.base.append(nn.MaxPool2D(kernel_size=2, stride=2))
              else:
                self.base.append ( DWConvLayer(ch, ch, stride=2) )
            
        
        ch = ch_list[blks-1]
        self.base.append (
            nn.Sequential(
                nn.AdaptiveAvgPool2D((1,1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, class_dim) ))
                
        #print(self.base)
        
        if pretrained:
          '''
          if hasattr(paddle, 'hub'):
          
            if arch == 68 and not depth_wise:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth'
            elif arch == 85 and not depth_wise:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet85-a28faa00.pth'
            elif arch == 68 and depth_wise:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth'
            else:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth'

            self.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
          else:
            postfix = 'ds' if depth_wise else ''
            weight_file = '%shardnet%d%s.pth'%(weight_path, arch, postfix)            
            if not os.path.isfile(weight_file):
              print(weight_file,'is not found')
              exit(0)
            weights = paddle.load(weight_file)
            self.load_state_dict(weights)
          '''
          postfix = 'DS' if depth_wise else ''
        
          print('ImageNet pretrained weights for HarDNet%d%s is loaded'%(arch, postfix))
          
    def forward(self, x):
        for layer in self.base:
          x = layer(x)
        return x
        
def hardnet68(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Harmonic DenseNet 68 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = HarDNet(depth_wise=False, arch=68, pretrained=pretrained)
    return model

def hardnet85(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Harmonic DenseNet 85 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = HarDNet(depth_wise=False, arch=85, pretrained=pretrained)
    return model

def hardnet68ds(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Harmonic DenseNet 68ds (Depthwise Separable) model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = HarDNet(depth_wise=True, arch=68, pretrained=pretrained)
    return model

def hardnet39ds(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Harmonic DenseNet 68ds (Depthwise Separable) model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = HarDNet(depth_wise=True, arch=39, pretrained=pretrained)
    return model
       
        
        
