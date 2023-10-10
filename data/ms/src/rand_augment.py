# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import random
from PIL import Image, ImageOps, ImageEnhance, ImageDraw


_MAX_LEVEL = 10.0
replace_values = (128, 128, 128)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def ShearX(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    v = _randomly_negate(v)
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0), Image.BICUBIC, fillcolor=replace_values)


def ShearY(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    v = _randomly_negate(v)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0), Image.BICUBIC, fillcolor=replace_values)


def TranslateX(img, v):  # [-0.45, 0.45]
    # assert -0.45 <= v <= 0.45
    v = _randomly_negate(v)
    v = int(v * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0), Image.BICUBIC, fillcolor=replace_values)


def TranslateXabs(img, v):  # [-100, 100]
    # assert v >= 0
    v = _randomly_negate(v)
    v = int(v)
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0), Image.BICUBIC, fillcolor=replace_values)


def TranslateY(img, v):  # [-0.45, 0.45]
    # assert -0.45 <= v <= 0.45
    v = _randomly_negate(v)
    v = int(v * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v), Image.BICUBIC, fillcolor=replace_values)


def TranslateYabs(img, v):  # [-100, 100]
    # assert v >= 0
    v = _randomly_negate(v)
    v = int(v)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v), Image.BICUBIC, fillcolor=replace_values)


def Rotate(img, v):  # [-30, 30]
    # assert -30 <= v <= 30
    v = _randomly_negate(v)
    return img.rotate(v)


def Solarize(img, v):  # [0, 255]
    # assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def SolarizeAdd(img, addition, threshold=128): # [0, 110]
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(min(255, i + addition))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def Posterize(img, v):  # [0, 4]
    # assert 0 <= v <= 4
    v = int(v)
    return ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 80] => percentage: [0, 0.35]
    # assert 0.0 <= v <= 0.35
    if v <= 0.:
        return img

    v = v * min(img.size[0], img.size[1])
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 80] => percentage: [0, 0.35]
    if v < 0:
        return img
    v = int(v)
    w, h = img.size
    x0 = random.uniform(0, w)
    y0 = random.uniform(0, h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, replace_values)
    return img


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Equalize(img, _):
    return ImageOps.equalize(img)


def augment_list():
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L510
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L691
    return [(AutoContrast, 0, 1),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Rotate, 0, 30),
            (Posterize, 0, 4),
            (Solarize, 0, 256),
            (Color, 0.1, 1.9),
            (Contrast, 0.1, 1.9),
            (Brightness, 0.1, 1.9),
            (Sharpness, 0.1, 1.9),
            (ShearX, 0.0, 0.3),
            (ShearY, 0.0, 0.3),
            (TranslateX, 0, 0.45),
            (TranslateY, 0, 0.45),
            (Cutout, 0, 0.35),
            (SolarizeAdd, 0, 110)]


class RandAugment:
    """RandAugment"""
    def __init__(self, num_layers, magnitude, magnitude_std=0.5):
        assert 0 <= magnitude and magnitude <= 30
        self.num_layers = num_layers
        self.magnitude = float(magnitude)
        self.magnitude_std = magnitude_std
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = []
        magnitude = random.gauss(self.magnitude, self.magnitude_std)

        for _ in range(self.num_layers):
            ops.append(random.choice(self.augment_list))
        for op, minval, maxval in ops:
            val = (magnitude / _MAX_LEVEL) * (maxval - minval) + minval
            img = op(img, val)

        return img
