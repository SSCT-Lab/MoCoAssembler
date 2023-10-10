import tensorflow as tf
import numpy as np

from model.customLayer import _SeparableConv, _Conv

def _TTFNetHeatmapHead(x, filters, **config_dict):
	prefix="HeatmapHead/"
	classNum = config_dict["classNum"]
	config_dict.pop('classNum')

	conf_dict_last = config_dict.copy()
	config_dict['use_bias'] = False
	config_dict.pop('bias_initializer')

	x=_SeparableConv(x, filters=filters, kernel_size=3, strides=1, prefix=prefix+"Sep/", **config_dict)
	x=_Conv(x, filters=classNum, kernel_size=1, strides=1, normalization=None, activation=None, prefix=prefix+"Conv/", **conf_dict_last)
	return x

def _TTFNetSizeHead(x, filters, **config_dict):
	prefix="SizeHead/"
	conf_dict_last = config_dict.copy()
	config_dict.pop('bias_initializer')
	config_dict['use_bias'] = False

	x=_SeparableConv(x, filters=filters, kernel_size=3, strides=1, prefix=prefix+"Sep/", **config_dict)
	x=_Conv(x, filters=4, kernel_size=1, strides=1, normalization=None, activation=None, prefix=prefix+"Conv/", **conf_dict_last)
	return x

def TTFNet(x, config=None):
	Size_config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["head"]["regularization"]),
        'kernel_initializer': tf.initializers.TruncatedNormal(mean=0.0, stddev=0.03),
        'trainable':not config["model_config"]["head"]["isFreeze"],
		'bias_initializer': tf.constant_initializer(-2.20),
        'use_bias':True,
    }

	Heatmap_config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["head"]["regularization"]),
        'kernel_initializer': tf.initializers.TruncatedNormal(mean=0.0, stddev=0.03),
        'bias_initializer': tf.constant_initializer(-4.6),
        'trainable':not config["model_config"]["head"]["isFreeze"], 
        'use_bias':True,
        'classNum':config["training_config"]["num_classes"],
    }

	Headmap_outputs=_TTFNetHeatmapHead(x, config["model_config"]["head"]["filters"], **Heatmap_config_dict)
	Size_outputs=_TTFNetSizeHead(x, config["model_config"]["head"]["filters"], **Size_config_dict)
	output = tf.keras.layers.Concatenate(axis=-1, name="LastConcat")([Headmap_outputs, Size_outputs])
	return tf.keras.layers.Activation('sigmoid', dtype='float32', name="output_layer")(output)
	
