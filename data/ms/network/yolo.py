import math
import numpy as np
from pathlib import Path
from copy import deepcopy
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import HeUniform
from utils.general import make_divisible
from utils.autoanchor import check_anchor_order
from network.common import parse_model, Detect, Segment


def initialize_weights(model):
    for m in model.cells():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t in (nn.BatchNorm2d, nn.SyncBatchNorm):
            m.eps = 1e-3
            m.momentum = 0.03

@ops.constexpr
def _get_h_w_list(ratio, gs, hw):
    return tuple([math.ceil(x * ratio / gs) * gs for x in hw])

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = ops.ResizeBilinear(size=s, align_corners=False)(img)
        if not same_shape:  # pad/crop img
            h, w = _get_h_w_list(ratio, gs, (h, w))

        # img = F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
        img = ops.pad(img, ((0, 0), (0, 0), (0, w - s[1]), (0, h - s[0])))
        img[:, :, -(w - s[1]):, :] = 0.447
        img[:, :, :, -(h - s[0]):] = 0.447
        return img

@ops.constexpr
def _get_stride_max(stride):
    return int(stride.max())

# class Model(nn.Cell):
#     def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, sync_bn=False):  # model, input channels, number of classes
#         super(Model, self).__init__()
#         self.traced = False
#         if isinstance(cfg, dict):
#             self.yaml = cfg  # model dict
#         else:  # is *.yaml
#             import yaml  # for torch hub
#             self.yaml_file = Path(cfg).name
#             with open(cfg) as f:
#                 self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

#         # Define model
#         ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
#         if nc and nc != self.yaml['nc']:
#             print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
#             self.yaml['nc'] = nc  # override yaml value
#         if anchors:
#             print(f'Overriding model.yaml anchors with anchors={anchors}')
#             self.yaml['anchors'] = round(anchors)  # override yaml value
#         self.model, self.save, self.layers_param = parse_model(deepcopy(self.yaml), ch=[ch], sync_bn=sync_bn)
#         self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
#         self.inplace = self.yaml.get('inplace', True)
#         # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

#         # Build strides, anchors
#         m = self.model[-1]  # Detect()
#         if isinstance(m, (Detect, Segment)):
#             s = 256  # 2x min stride
#             m.inplace = self.inplace
#             # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
#             m.stride = Tensor(np.array(self.yaml['stride']), ms.int32)
#             check_anchor_order(m)
#             m.anchors /= m.stride.view(-1, 1, 1)
#             self.stride = m.stride
#             self.stride_np = np.array(self.yaml['stride'])
#             self._initialize_biases()  # only run once
#             # print('Strides: %s' % m.stride.tolist())

#         # Init weights, biases
#         initialize_weights(self)

#     # @ms.ms_function
#     def construct(self, x, augment=False):
#         if augment:
#             img_size = x.shape[-2:]  # height, width
#             s = (1, 0.83, 0.67)  # scales
#             f = (None, 3, None)  # flips (2-ud, 3-lr)
#             y = ()  # outputs
#             for si, fi in zip(s, f):
#                 xi = scale_img(ops.ReverseV2(fi)(x) if fi else x, si, gs=_get_stride_max(self.stride_np))
#                 # xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
#                 yi = self.forward_once(xi)[0]  # forward
#                 # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
#                 yi[..., :4] /= si  # de-scale
#                 if fi == 2:
#                     yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
#                 elif fi == 3:
#                     yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
#                 y += (yi,)
#             return ops.concat(y, 1) # augmented inference, train
#         else:
#             return self.forward_once(x)  # single-scale inference, train

#     def forward_once(self, x):
#         y, dt = (), ()  # outputs
#         for i in range(len(self.model)):
#             m = self.model[i]
#             iol, f, _, _ = self.layers_param[i] # iol: index of layers

#             if not(isinstance(f, int) and f == -1): # if not from previous layer
#                 # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#                 if isinstance(f, int):
#                     x = y[f]
#                 else:
#                     _x = ()
#                     for j in f:
#                         if j == -1:
#                             _x += (x,)
#                         else:
#                             _x += (y[j],)
#                     x = _x

#             if self.traced:
#                 if isinstance(m, Detect):
#                     break

#             # print("index m: ", iol) # print if debug on pynative mode, not available on graph mode.
#             x = m(x)  # run

#             y += (x if iol in self.save else None,)  # save output

#         return x

#     def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
#         # https://arxiv.org/abs/1708.02002 section 3.3
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
#         m = self.model[-1]  # Detect() module
#         for mi, s in zip(m.m, m.stride):  # from
#             s = s.asnumpy()
#             b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
#             b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#             b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
#             mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))

class Model(nn.Cell):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None, sync_bn=False, opt=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.layers_param = parse_model(deepcopy(self.yaml), ch=[ch], sync_bn=sync_bn)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Recompute
        if opt is not None:
            if opt.recompute and opt.recompute_layers > 0:
                for i in range(opt.recompute_layers):
                    self.model[i].recompute()
                print(f"Turn on recompute, and the results of the first {opt.recompute_layers} layers "
                      f"will be recomputed.")

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = Tensor(np.array(self.yaml['stride']), ms.int32)
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.stride_np = np.array(self.yaml['stride'])
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self.model)

    def construct(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = (1, 0.83, 0.67)  # scales
            f = (None, 3, None)  # flips (2-ud, 3-lr)
            y = ()  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(ops.ReverseV2(fi)(x) if fi else x, si, gs=_get_stride_max(self.stride_np))
                # xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y += (yi,)
            return ops.concat(y, 1) # augmented inference, train
        else:
            return self.forward_once(x)  # single-scale inference, train

    def forward_once(self, x):
        y, dt = (), ()  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            iol, f, _, _ = self.layers_param[i] # iol: index of layers

            if not(isinstance(f, int) and f == -1): # if not from previous layer
                # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(f, int):
                    x = y[f]
                else:
                    _x = ()
                    for j in f:
                        if j == -1:
                            _x += (x,)
                        else:
                            _x += (y[j],)
                    x = _x

            if self.traced:
                if isinstance(m, Detect):
                    break

            # print("index m: ", iol) # print if debug on pynative mode, not available on graph mode.
            x = m(x)  # run

            y += (x if iol in self.save else None,)  # save output

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            s = s.asnumpy()
            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))



if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    cfg = "./config/models/yolov5s.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=None)
    for p in model.trainable_params():
        print(p.name)
    # model.set_train(True)
    # x = Tensor(np.random.randn(1, 3, 160, 160), ms.float32)
    # pred = model(x)
    # print(pred)