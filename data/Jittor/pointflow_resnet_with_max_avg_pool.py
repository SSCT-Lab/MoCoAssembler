import jittor as jt
from jittor import nn
from network.nn.operators import PSPModule
from network.nn.point_flow import PointFlowModuleWithMaxAvgpool
from network import resnet_d as Resnet_Deep
from jittor.models.res2net import res2net101_26w_4s


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride), normal_layer(out_planes), nn.ReLU())


class UperNetAlignHeadMaxAvgpool(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256, fpn_dsn=False, reduce_dim=64, ignore_background=False, max_pool_size=8, avgpool_size=8, edge_points=32):
        super(UperNetAlignHeadMaxAvgpool, self).__init__()
        self.ppm = PSPModule(
            inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:(- 1)]:
            self.fpn_in.append(nn.Sequential(
                nn.Conv(fpn_inplane, fpn_dim, 1), norm_layer(fpn_dim), nn.ReLU()))
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range((len(fpn_inplanes) - 1)):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))

            self.fpn_out_align.append(PointFlowModuleWithMaxAvgpool(
                fpn_dim, dim=reduce_dim, maxpool_size=max_pool_size, avgpool_size=avgpool_size, edge_points=edge_points))
            if self.fpn_dsn:
                self.dsn.append(nn.Sequential(nn.Conv(fpn_dim, fpn_dim, 3, stride=1, padding=1), norm_layer(
                    fpn_dim), nn.ReLU(), nn.Dropout(p=0.1), nn.Conv(fpn_dim, num_class, 1, stride=1, padding=0, bias=True)))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)
        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)
        self.conv_last = nn.Sequential(conv3x3_bn_relu(
            (len(fpn_inplanes) * fpn_dim), fpn_dim, 1), nn.Conv(fpn_dim, num_class, 1))

    def execute(self, conv_out):
        psp_out = self.ppm(conv_out[-1])
        f = psp_out
        fpn_feature_list = [f]
        edge_preds = []
        out = []
        for i in reversed(range((len(conv_out) - 1))):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            (f, edge_pred) = self.fpn_out_align[i]([f, conv_x])
            f = (conv_x + f)
            edge_preds.append(edge_pred)
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))
        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].shape[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.interpolate(
                fpn_feature_list[i], output_size, mode='bilinear', align_corners=True))
        fusion_out = jt.contrib.concat(fusion_list, dim=1)
        x = self.conv_last(fusion_out)
        return (x, edge_preds)


class AlignNetResNetMaxAvgpool(nn.Module):

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, variant='D', skip='m1', skip_num=48, fpn_dsn=False, inplanes=128, reduce_dim=64, ignore_background=False, max_pool_size=8, avgpool_size=8, edge_points=32):
        super(AlignNetResNetMaxAvgpool, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn
        if (trunk == trunk == 'resnet-50-deep'):
            resnet = Resnet_Deep.resnet50()
        elif (trunk == 'resnet-101-deep'):
            resnet = Resnet_Deep.resnet101()
        elif (trunk == 'res2net-101-deep'):
            resnet = res2net101_26w_4s(pretrained=True)
        else:
            raise ValueError('Not a valid network arch')
        resnet.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        (self.layer1, self.layer2, self.layer3, self.layer4) = (
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        del resnet
        if (self.variant == 'D'):
            for (n, m) in self.layer3.named_modules():
                if ('conv2' in n):
                    (m.dilation, m.padding) = ((2, 2), (2, 2))
                elif ('downsample.0' in n):
                    m.stride = (2, 2)
            for (n, m) in self.layer4.named_modules():
                if ('conv2' in n):
                    (m.dilation, m.padding) = ((4, 4), (4, 4))
                elif ('downsample.0' in n):
                    m.stride = (2, 2)
        else:
            print('Not using Dilation ')
        inplane_head = 2048
        self.head = UperNetAlignHeadMaxAvgpool(inplane_head, num_class=num_classes, norm_layer=nn.BatchNorm2d, fpn_dsn=fpn_dsn, reduce_dim=reduce_dim,
                                               ignore_background=ignore_background, max_pool_size=max_pool_size, avgpool_size=avgpool_size, edge_points=edge_points)

    def execute(self, x, gts=None):
        x_size = x.shape
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.head([x1, x2, x3, x4])
        main_out = nn.interpolate(
            x[0], size=x_size[2:], mode='bilinear', align_corners=True)

        edge_preds = [nn.interpolate(
            edge_pred, size=x_size[2:], mode='bilinear', align_corners=True) for edge_pred in x[1]]
        if self.is_training():
            if not self.fpn_dsn:
                return self.criterion([main_out, edge_preds], gts)
            return self.criterion(x, gts)
        return main_out


def DeepR101_PF_maxavg_deeply(num_classes, criterion, reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32):
    return AlignNetResNetMaxAvgpool(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1', reduce_dim=reduce_dim, max_pool_size=max_pool_size, avgpool_size=avgpool_size, edge_points=edge_points)


def DeepR50_PF_maxavg_deeply(num_classes, criterion, reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32):
    return AlignNetResNetMaxAvgpool(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1', reduce_dim=reduce_dim, max_pool_size=max_pool_size, avgpool_size=avgpool_size, edge_points=edge_points)


def DeepR2N101_PF_maxavg_deeply(num_classes, criterion, reduce_dim=64, max_pool_size=8, avgpool_size=8, edge_points=32):
    return AlignNetResNetMaxAvgpool(num_classes, trunk='res2net-101-deep', criterion=criterion, variant='D', skip='m1', reduce_dim=reduce_dim, max_pool_size=max_pool_size, avgpool_size=avgpool_size, edge_points=edge_points)