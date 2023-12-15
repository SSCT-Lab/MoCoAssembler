class Filter:
    def __init__(self):
        self.info_lis = []
        self.addkey("must be divisible by groups")
        self.addkey("pad should be at most half of kernel size")
        self.addkey("MaxUnpool2d.forward() missing 1 required positional argument")
        self.addkey("Kernel size can't be greater than actual input size")
        self.addkey("out_channels must be divisible by groups")
        self.addkey("but got Tensor of dimension")
        self.addkey("mat1 and mat2 shapes cannot be multiplied")
        self.addkey("Bilinear.forward() missing 1 required positional argument")
        self.addkey("No module named")
        self.addkey("but got input of size")

    def judge(self, string) -> bool:
        for s in self.info_lis:
            if s in string:
                return False
            else:
                continue
        return True

    def addkey(self, string: str) -> None:
        self.info_lis.append(string)
