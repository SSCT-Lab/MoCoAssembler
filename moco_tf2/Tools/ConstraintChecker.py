from abc import ABC, abstractmethod

from moco_tf2.Utils.utils import get_max_number


class Checker(ABC):
    @abstractmethod
    def check(self, block):
        # check a block
        pass


class tf_keras_layer_Conv1D_Checker(Checker):
    def check(self, block):
        strides = block.get_param_value("strides", 1)
        strides = strides if isinstance(strides, int) else get_max_number(strides)

        dilation_rate = block.get_param_value("dilation_rate", 1)
        dilation_rate = dilation_rate if isinstance(dilation_rate, int) else get_max_number(dilation_rate)

        groups = block.get_param_value("groups", 1)
        groups = groups if isinstance(groups, int) else int(groups)

        filters = block.get_param_value("filters", 1)
        filters = filters if isinstance(filters, int) else int(filters)

        flags = [
            filters % groups == 0,
            strides == 1 or dilation_rate == 1
        ]

        flag = True
        for f in flags:
            flag = f and flag
        return flag


class tf_keras_layer_Conv2D_Checker(Checker):
    def check(self, block):
        strides = block.get_param_value("strides", 1)
        strides = strides if isinstance(strides, int) else get_max_number(strides)

        dilation_rate = block.get_param_value("dilation_rate", 1)
        dilation_rate = dilation_rate if isinstance(dilation_rate, int) else get_max_number(dilation_rate)

        groups = block.get_param_value("groups", 1)
        groups = groups if isinstance(groups, int) else int(groups)

        filters = block.get_param_value("filters", 1)
        filters = filters if isinstance(filters, int) else int(filters)

        flags = [
            filters % groups == 0,
            strides == 1 or dilation_rate == 1
        ]

        flag = True
        for f in flags:
            flag = f and flag
        return flag


class tf_keras_layer_Conv3D_Checker(Checker):
    def check(self, block):
        strides = block.get_param_value("strides", 1)
        strides = strides if isinstance(strides, int) else get_max_number(strides)

        dilation_rate = block.get_param_value("dilation_rate", 1)
        dilation_rate = dilation_rate if isinstance(dilation_rate, int) else get_max_number(dilation_rate)

        groups = block.get_param_value("groups", 1)
        groups = groups if isinstance(groups, int) else int(groups)

        filters = block.get_param_value("filters", 1)
        filters = filters if isinstance(filters, int) else int(filters)

        flags = [
            filters % groups == 0,
            strides == 1 or dilation_rate == 1
        ]

        flag = True
        for f in flags:
            flag = f and flag
        return flag


class tf_keras_layers_Conv1DTranspose_Checker(Checker):
    def check(self, block):
        strides = block.get_param_value("strides", 1)
        strides = strides if isinstance(strides, int) else get_max_number(strides)

        dilation_rate = block.get_param_value("dilation_rate", 1)
        dilation_rate = dilation_rate if isinstance(dilation_rate, int) else get_max_number(dilation_rate)

        groups = block.get_param_value("groups", 1)
        groups = groups if isinstance(groups, int) else int(groups)

        filters = block.get_param_value("filters", 1)
        filters = filters if isinstance(filters, int) else int(filters)

        output_padding = block.get_param_value("output_padding", strides - 1)
        output_padding = output_padding if isinstance(output_padding, int) else int(output_padding)

        flags = [
            filters % groups == 0,
            strides == 1 or dilation_rate == 1,
            output_padding < strides
        ]

        flag = True
        for f in flags:
            flag = f and flag
        return flag


class tf_keras_layers_Conv2DTranspose_Checker(Checker):
    def check(self, block):
        strides = block.get_param_value("strides", 1)
        strides = strides if isinstance(strides, int) else get_max_number(strides)

        dilation_rate = block.get_param_value("dilation_rate", 1)
        dilation_rate = dilation_rate if isinstance(dilation_rate, int) else get_max_number(dilation_rate)

        groups = block.get_param_value("groups", 1)
        groups = groups if isinstance(groups, int) else int(groups)

        filters = block.get_param_value("filters", 1)
        filters = filters if isinstance(filters, int) else int(filters)

        output_padding = block.get_param_value("output_padding", strides - 1)
        output_padding = output_padding if isinstance(output_padding, int) else int(output_padding)

        flags = [
            filters % groups == 0,
            strides == 1 or dilation_rate == 1,
            output_padding < strides
        ]

        flag = True
        for f in flags:
            flag = f and flag
        return flag


class tf_keras_layers_Conv3DTranspose_Checker(Checker):
    def check(self, block):
        strides = block.get_param_value("strides", 1)
        strides = strides if isinstance(strides, int) else get_max_number(strides)

        dilation_rate = block.get_param_value("dilation_rate", 1)
        dilation_rate = dilation_rate if isinstance(dilation_rate, int) else get_max_number(dilation_rate)

        groups = block.get_param_value("groups", 1)
        groups = groups if isinstance(groups, int) else int(groups)

        filters = block.get_param_value("filters", 1)
        filters = filters if isinstance(filters, int) else int(filters)

        output_padding = block.get_param_value("output_padding", strides - 1)
        output_padding = output_padding if isinstance(output_padding, int) else int(output_padding)

        flags = [
            filters % groups == 0,
            strides == 1 or dilation_rate == 1,
            output_padding < strides
        ]

        flag = True
        for f in flags:
            flag = f and flag
        return flag


checkers = {
    "tf.keras.layers.Conv1D":tf_keras_layer_Conv1D_Checker,
    "tf.keras.layers.Convolution1D":tf_keras_layer_Conv1D_Checker,
    "tf.keras.layers.ConvLSTM1D":tf_keras_layer_Conv1D_Checker,
    "tf.keras.layers.DepthwiseConv1D":tf_keras_layer_Conv1D_Checker,
    "tf.keras.layers.SeparableConv1D":tf_keras_layer_Conv1D_Checker,

    "tf.keras.layers.Conv2D": tf_keras_layer_Conv2D_Checker,
    "tf.keras.layers.Convolution2D": tf_keras_layer_Conv2D_Checker,
    "tf.keras.layers.ConvLSTM2D": tf_keras_layer_Conv2D_Checker,
    "tf.keras.layers.DepthwiseConv2D": tf_keras_layer_Conv2D_Checker,
    "tf.keras.layers.SeparableConv2D": tf_keras_layer_Conv2D_Checker,

    "tf.keras.layers.Conv3D": tf_keras_layer_Conv3D_Checker,
    "tf.keras.layers.Convolution3D": tf_keras_layer_Conv3D_Checker,
    "tf.keras.layers.ConvLSTM3D": tf_keras_layer_Conv3D_Checker,

    "tf.keras.layers.Conv1DTranspose":tf_keras_layers_Conv1DTranspose_Checker,
    "tf.keras.layers.Conv2DTranspose": tf_keras_layers_Conv2DTranspose_Checker,
    "tf.keras.layers.Conv3DTranspose": tf_keras_layers_Conv3DTranspose_Checker
}


def check_block(block):
    if block.api_name in checkers.keys():
        return checkers[block.api_name]().check(block)
    else:
        return True
