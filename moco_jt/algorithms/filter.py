class Filter:
    def __init__(self):
        self.info_lis = []
        self.addkey('nn.py", line 31, in matmul_transpose')  # shape problem
        self.addkey('nn.py", line 962')  # strange problem
        self.addkey('nn.py", line 957, in execute')  # Conv dim problem
        self.addkey('nn.py", line 959, in execute')  # Conv channels problem
        self.addkey('Please refer to examples(help(jt.ops.reshape))')  # reshape problem
        self.addkey('nn.py", line 1369, in execute')  # ConvTranspose dim problem
        self.addkey('not enough values to unpack')  # Other dim problems
        self.addkey('too many values to unpack')  # Other dim problems
        self.addkey(') Shape not match, x:')  # Norm Layers shape problem
        self.addkey('need_sync->num >= 0')  # some -1 shape problem
        self.addkey('Linear(in_features = -1')  # Linear shape problem: in_features
        self.addkey('input channel needs to be divided by upscale_factor')  # PixelShuffle shape Problem
        self.addkey('ValueError: math domain error')  # some -1 shape problem
        self.addkey('No module named')  # some questions about module import...
        self.addkey('MaxUnpool2d.execute() missing 1 required positional argument:')  # as it said
        self.addkey('out_channels must be divisible by groups')  # as it said

        # level 2 filter: checked errors

        # self.addkey('AttributeError: module \'jittor\' has no attribute \'softplus\'')  # Mish
        # self.addkey('nn.py", line 2160, in execute')  # UpSample
        # self.addkey('nn.py", line 2163, in execute')  # Another UpSample
        # self.addkey('maximun')  # AdaptiveAvgPool3d
        # self.addkey(':1:1:1:i1:o1:s0,')  # MaxPool2d Crush
        # self.addkey('Vary shape should only occur in the first dimension')  # MaxPool2d CYPRTI

    def judge(self, string) -> bool:  # 如果含有以上字符串，那将被过滤掉，返回False，如果是有价值信息，返回True
        for s in self.info_lis:
            if s in string:
                return False
            else:
                continue
        return True

    def addkey(self, string: str) -> None:
        self.info_lis.append(string)
