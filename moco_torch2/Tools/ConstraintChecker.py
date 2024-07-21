from abc import ABC, abstractmethod


class Checker(ABC):
    @abstractmethod
    def check(self, block):
        # check a block
        pass


class torch_nn_Conv1d_Checker(Checker):
    def check(self, block):
        groups = block.GetParamValue("groups", 1)
        in_channels = block.GetParamValue("in_channels", block.inChannels)
        out_channels = block.GetParamValue("out_channels", block.outChannels)
        padding = block.GetParamValue("padding", 0)
        stride = block.GetParamValue("stride", 0)
        flags = [
            in_channels % groups == 0,
            out_channels % groups == 0,
            groups <= in_channels,
            groups <= out_channels,
            (((groups == in_channels) and (out_channels % in_channels == 0)) or (not (groups == in_channels))),
            (((padding == 'same') and (stride == 1)) or (not (padding == 'same')))
        ]
        flag = True
        for f in flags:
            flag = f and flag
        return flag


class torch_nn_Conv2d_Checker(Checker):
    def check(self, block):
        groups = block.GetParamValue("groups", 1)
        in_channels = block.GetParamValue("in_channels", block.inChannels)
        out_channels = block.GetParamValue("out_channels", block.outChannels)
        padding = block.GetParamValue("padding", 0)
        stride = block.GetParamValue("stride", 0)
        flags = [
            in_channels % groups == 0,
            out_channels % groups == 0,
            groups <= in_channels,
            groups <= out_channels,
            (((groups == in_channels) and (out_channels % in_channels == 0)) or (not (groups == in_channels))),
            (((padding == 'same') and (stride == 1)) or (not (padding == 'same')))
        ]
        flag = True
        for f in flags:
            flag = f and flag
        return flag


class torch_nn_Conv3d_Checker(Checker):
    def check(self, block):
        groups = block.GetParamValue("groups", 1)
        in_channels = block.GetParamValue("in_channels", block.inChannels)
        out_channels = block.GetParamValue("out_channels", block.outChannels)
        padding = block.GetParamValue("padding", 0)
        stride = block.GetParamValue("stride", 0)
        flags = [
            in_channels % groups == 0,
            out_channels % groups == 0,
            groups <= in_channels,
            groups <= out_channels,
            (((groups == in_channels) and (out_channels % in_channels == 0)) or (not (groups == in_channels))),
            (((padding == 'same') and (stride == 1)) or (not (padding == 'same')))
        ]
        flag = True
        for f in flags:
            flag = f and flag
        return


class torch_nn_ConvTranspose1d_Checker(Checker):
    def check(self, block):
        groups = block.GetParamValue("groups", 1)
        in_channels = block.GetParamValue("in_channels", block.inChannels)
        out_channels = block.GetParamValue("out_channels", block.outChannels)
        flags = [
            in_channels % groups == 0,
            out_channels % groups == 0,
            groups <= in_channels,
            groups <= out_channels
        ]
        flag = True
        for f in flags:
            flag = f and flag
        return flag


class torch_nn_ConvTranspose2d_Checker(Checker):
    def check(self, block):
        groups = block.GetParamValue("groups", 1)
        in_channels = block.GetParamValue("in_channels", block.inChannels)
        out_channels = block.GetParamValue("out_channels", block.outChannels)
        flags = [
            in_channels % groups == 0,
            out_channels % groups == 0,
            groups <= in_channels,
            groups <= out_channels
        ]
        flag = True
        for f in flags:
            flag = f and flag
        return flag


class torch_nn_ConvTranspose3d_Checker(Checker):
    def check(self, block):
        groups = block.GetParamValue("groups", 1)
        in_channels = block.GetParamValue("in_channels", block.inChannels)
        out_channels = block.GetParamValue("out_channels", block.outChannels)
        flags = [
            in_channels % groups == 0,
            out_channels % groups == 0,
            groups <= in_channels,
            groups <= out_channels
        ]
        flag = True
        for f in flags:
            flag = f and flag
        return flag


checkers = {
    "torch.nn.Conv1d": torch_nn_Conv1d_Checker,
    "torch.nn.Conv2d": torch_nn_Conv2d_Checker,
    "torch.nn.Conv3d": torch_nn_Conv3d_Checker,
    "torch.nn.ConvTranspose1d": torch_nn_ConvTranspose1d_Checker,
    "torch.nn.ConvTranspose2d": torch_nn_ConvTranspose2d_Checker,
    "torch.nn.ConvTranspose3d": torch_nn_ConvTranspose3d_Checker
}


def CheckBlock(block):
    if block.apiName in checkers.keys():
        return checkers[block.apiName]().check(block)
    else:
        return True
