import utils
utils.set_evironments(seed=212)
# Tensorflow
import tensorflow as tf



###########################################################################
#                                Network
###########################################################################


# 设定初始化
KERNEL_INITIALIZER = 'he_normal'
KERNEL_REGULARUZER = tf.keras.regularizers.l2(5e-5)



class SwichActivation(tf.keras.layers.Layer):    
	def __init__(self, ):
		super(SwichActivation, self).__init__()

	def call(self, x):
		return tf.keras.backend.sigmoid(x) * x



class InvertedResidualBlock(tf.keras.layers.Layer):
	def __init__(self, filters_in=32,  filters_out=16, kernel_size=3, strides=1, expand_ratio=1, se_ratio=0.25, name="stage2"):
		super(InvertedResidualBlock, self).__init__()

		filters = filters_in * expand_ratio

		self.conv_layer1 = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(filters, 1, use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER),
			tf.keras.layers.BatchNormalization(),
			SwichActivation(),
			tf.keras.layers.DepthwiseConv2D(kernel_size, strides, padding='same', use_bias=False, depthwise_initializer=KERNEL_INITIALIZER, depthwise_regularizer=KERNEL_REGULARUZER),
			tf.keras.layers.BatchNormalization(),
			SwichActivation()
		])

		filters_se = max(1, int(filters_in * se_ratio))

		self.se_conv = tf.keras.models.Sequential([
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Reshape((1, 1, filters)),
			tf.keras.layers.Conv2D(filters_se, 1, use_bias=False,  kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER),
			SwichActivation(),
			tf.keras.layers.Conv2D(filters, 1, use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER),
			tf.keras.layers.Activation('sigmoid')
		])

		self.multiply = tf.keras.layers.Multiply()

		self.conv_layer2 = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(filters_out, 1, use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER),
			tf.keras.layers.BatchNormalization()
		])


	def call(self, x, training=None):
		x = self.conv_layer1(x)
		se = self.se_conv(x)
		x = self.multiply([x, se])
		x = self.conv_layer2(x)
		return x



# 官方提供的 backbone 模型

class GEN(tf.keras.layers.Layer):   
	
	def __init__(self, **kwargs):
		super(GEN, self).__init__()

		self.conv1 = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(16, 5, padding='same', use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER),
			tf.keras.layers.BatchNormalization(),
			SwichActivation()
		])

		self.blocks = tf.keras.models.Sequential([
			InvertedResidualBlock(16, 24, 5, 2, 6, name='stage2'),
			InvertedResidualBlock(24, 40, 5, 2, 6, name='stage3'),
			InvertedResidualBlock(40, 80, 5, 2, 6, name='stage4'),
			InvertedResidualBlock(80, 40, 5, 1, 6, name='stage5'),
		])

		self.conv2 =  tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(768, 1, use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER, name='GEN_stage6_conv'),
			tf.keras.layers.BatchNormalization(),
			SwichActivation(),
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Dense(768, use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER),
			tf.keras.layers.Activation('sigmoid'),
			tf.keras.layers.Reshape((256, 3))
		])



	def call(self, inputs, training=False):

		x = self.conv1(inputs)
		x = self.blocks(x)
		x = self.conv2(x)

		return x
	




class LEN(tf.keras.Model):
	def __init__(self, **kwargs):
		super(LEN, self).__init__()

		self.first_conv = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(16, 5, padding='same', use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Activation('swish')
		])

		self.down_block_1 = InvertedResidualBlock(16, 24, 5, 2, 6)
		self.down_block_2 = InvertedResidualBlock(24, 40, 5, 2, 6)
		self.down_block_3 = InvertedResidualBlock(40, 80, 5, 2, 6)
		self.down_block_4 = InvertedResidualBlock(80, 40, 5, 1, 6)


		self.up_1 = tf.keras.layers.UpSampling2D(interpolation='bilinear')
		self.cat_1 = tf.keras.layers.Concatenate(axis=-1)

		self.up_block_1 = InvertedResidualBlock(80, 24, 5, 1, 6)

		self.up_2 = tf.keras.layers.UpSampling2D(interpolation='bilinear')
		self.cat_2 = tf.keras.layers.Concatenate(axis=-1)

		self.up_block_2 = InvertedResidualBlock(48, 16, 5, 1, 6)

		self.up_3 = tf.keras.layers.UpSampling2D(interpolation='bilinear')
		self.cat_3 = tf.keras.layers.Concatenate(axis=-1)

		self.last_conv = tf.keras.layers.Conv2D(3, 5, padding='same', use_bias=False, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=KERNEL_REGULARUZER,)
		
		self.add = tf.keras.layers.Add()


	def call(self, inputs):

		x1 = self.first_conv(inputs)

		x2 = self.down_block_1(x1)
		x3 = self.down_block_2(x2)
		x4 = self.down_block_3(x3)
		x4 = self.down_block_4(x4)
		
		x4 = self.up_1(x4)
		x3 = self.cat_1([x3, x4])
		x3 = self.up_block_1(x3)

		x3 = self.up_2(x3)
		x2 = self.cat_2([x2, x3])
		x2 = self.up_block_2(x2)

		x2 = self.up_3(x2)
		x1 = self.cat_3([x1, x2])

		x1 = self.last_conv(x1)

		res = self.add([inputs, x1])

		return res	




class IntensitiyTransform(tf.keras.layers.Layer):    

	def __init__(self, channels, intensities, **kwargs):
		super(IntensitiyTransform, self).__init__(**kwargs)
		self.channels = channels
		self.scale = intensities - 1

	def call(self, inputs):
		im, it = inputs
		x = tf.map_fn(self._intensity_transform, [im, it], dtype='float32')
		return x
	
	def _intensity_transform(self, inputs):
		im, it = inputs
		im = tf.cast(tf.math.round(self.scale * im), dtype='int32')
		im = tf.split(im, num_or_size_splits=self.channels, axis=-1)
		it = tf.split(it, num_or_size_splits=self.channels, axis=-1)
		x = tf.concat([tf.gather_nd(a, b) for a, b in zip(it, im)], axis=-1)
		return x







	
	


class GleNet(tf.keras.Model):
	def __init__(self, backbone='GEN', low_size=[256, 256], use_local=False, **kwargs):
		super(GleNet, self).__init__()

		self.down_sampler = tf.keras.layers.Lambda(tf.image.resize, arguments={'size': low_size})
		
		# 求解 3 x 256 个值
		self.regressor = eval(backbone)(**kwargs)
		# 可以反向传播的映射
		self.curve_enhancer = IntensitiyTransform(3, 256)

		self.use_local = use_local

		if(self.use_local == True):
			self.refine = LEN()
			print('使用 LEN 局部增强网络')

			

	def call(self, inputs, is_training=True):

		if(is_training == False):
			x = self.down_sampler(inputs)
		else:
			x = inputs

		x = self.regressor(x)

		x = self.curve_enhancer([inputs, x])

		if(self.use_local):
			refined = self.refine(x)
		else: 
			refined = x + inputs

		return tf.clip_by_value(refined, 0, 1)



if __name__ == '__main__':


	
	network = GleNet(residual=True)

	network.build(input_shape=(4, 216, 256, 3))