import re


class Filter:
    def __init__(self):
        self.info_lis = []
        self._add_key(r".*?expected.*?found.*?")
        self._add_key(r".*?Negative dimension size caused by subtracting.*?")
        self._add_key(r".*?Argument `cropping` must be greater than the input shape.*?")
        self._add_key(r".*?Attention layer must be called on a list of inputs.*?")
        self._add_key(r".*?layer should be called on a list of.*?")
        self._add_key(r".*?One of the dimensions in the output is <= 0.*?")
        self._add_key(r".*?missing 1 required positional argument: 'states'.*?")
        self._add_key(r".*?`dim` must be in the range.*?")
        self._add_key(r".*?rank.*?")
        self._add_key(r".*?Strides must be greater than output padding.*?")
        self._add_key(r".*?`strides > 1` not supported in conjunction with `dilation_rate > 1`.*?")
        self._add_key(r".*?'images' must have either.*? or .*? dimensions.*?")
        self._add_key(r".*?`interpolation` argument should be one of.*?")
        self._add_key(r".*?list index out of range.*?")
        self._add_key(r".*?The number of input channels must be evenly divisible by the number of groups.*?")
        self._add_key(r".*?The number of filters must be evenly divisible by the number of groups.*?")
        self._add_key(r".*?`padding` should have (two|3) elements.*?")
        self._add_key(r".*?`padding` should be either an int, a tuple of (2|3) ints")
        self._add_key(r".*?The argument `(kernel_size|strides)` cannot contain 0(s).*?")
        self._add_key(r".*?The channel dimension of the inputs should be defined.*?")
        self._add_key(r".*?Inputs should have rank /d. Received input shape:.*?")

    def judge(self, string) -> bool:
        for s in self.info_lis:
            if len(re.findall(s, string)) > 0:
                return False
        return True

    def _add_key(self, string: str) -> None:
        self.info_lis.append(string)
