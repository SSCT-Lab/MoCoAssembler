import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
 
#（1）标准卷积模块
def conv_block(input_tensor, filters, kernel_size, stride):
 
    # 普通卷积+标准化+激活函数
    x = layers.Conv2D(filters = filters,  # 输出特征图个数
                      kernel_size = kernel_size,  # 卷积size
                      strides = stride,  # 步长
                      padding = 'same',  # 步长=1输出特征图size不变，步长=2特征图长宽减半
                      use_bias = False)(input_tensor)  # 有BN层就不需要偏置
    
    x = layers.BatchNormalization()(x)  # 批标准化
 
    x = layers.ReLU()(x)  # relu激活函数
 
    return x  # 返回标准卷积的输出特征图
 
 
#（2）深度可分离卷积模块
def sep_conv_block(input_tensor, filters, kernel_size):
 
    # 激活函数
    x = layers.ReLU()(input_tensor)
 
    # 深度可分离卷积函数，包含了（深度卷积+逐点卷积）
    x = layers.SeparableConvolution2D(filters = filters,  # 逐点卷积的卷积核个数，输出特征图个数
                                      kernel_size = kernel_size,  # 深度卷积的卷积核size
                                      strides = 1,  # 深度卷积的步长
                                      padding = 'same',  # 卷积过程中特征图size不变
                                      use_bias = False)(x)  # 有BN层就不要偏置
    
    return x  # 返回输出特征图
 
 
#（3）一个残差单元
def res_block(input_tensor, filters):
 
    # ① 残差边
    residual = layers.Conv2D(filters,  # 输出图像的通道数
                              kernel_size = (1,1),  # 卷积核size
                              strides = 2)(input_tensor)  # 使输入和输出的size相同
 
    residual = layers.BatchNormalization()(residual)  # 批标准化
 
    # ② 卷积块
    x = sep_conv_block(input_tensor, filters, kernel_size=(3,3))
    x = sep_conv_block(x, filters, kernel_size=(3,3))
    x = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
 
    # ③ 输入输出叠加，残差连接
    output = layers.Add()([residual, x])
 
    return output
 
 
#（4）Middle Flow模块
def middle_flow(x, filters):
 
    # 该模块循环8次
    for _ in range(8): 
 
        # 残差边
        residual = x
        # 三个深度可分离卷积块
        x = sep_conv_block(x, filters, kernel_size=(3,3))
        x = sep_conv_block(x, filters, kernel_size=(3,3))
        x = sep_conv_block(x, filters, kernel_size=(3,3))
        # 叠加残差边
        x = layers.Add()([residual, x])
 
    return x
 
 
#（5）主干网络
def xception(input_shape, classes):
 
    # input
    inputs = keras.Input(shape=input_shape)
    
    # 1st block
    # [299,299,3]==>[149,149,32]
    x = conv_block(inputs, filters=32, kernel_size=(3,3), stride=2)  # 标准卷积块
    
    # 2nd block
    # [149,149,32]==>[149,149,64]
    x = conv_block(x, filters=64, kernel_size=(3,3), stride=1)
    
    # 3rd block
    # [149,149,64]==>[75,75,128]
    # 残差边
    residual = layers.Conv2D(filters=128, kernel_size=(1,1), strides=2, 
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    # 4th block
    # 卷积块[149,149,64]==>[149,149,128]
    x = layers.SeparableConv2D(128, kernel_size=(3,3), strides=1, padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # 5th block
    # [149,149,128]==>[149,149,128]
    x = sep_conv_block(x, filters=128, kernel_size=(3,3))
    # [149,149,128]==>[75,75,128]
    x = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
    
    # 6th block
    # [75,75,128]==>[38,38,256]
    x = res_block(x, filters=256)

    # 7th block
    # [38,38,256]==>[19,19,728]
    x = res_block(x, filters=728)

    # 8th block
    # [19,19,728]==>[19,19,728]
    x = middle_flow(x, filters=728)
    
    # 9th block
    # 残差边模块[19,19,728]==>[10,10,1024]
    residual = layers.Conv2D(filters=1024, kernel_size=(1,1), 
                             strides=2, use_bias=False, padding='same')(x) 
    residual = layers.BatchNormalization()(residual)  # 批标准化
 
    # 10th block
    # 卷积块[19,19,728]==>[19,19,728]
    x = sep_conv_block(x, filters=728, kernel_size=(3,3))

    # 11th block
    # [19,19,728]==>[19,19,1024]
    x = sep_conv_block(x, filters=1024, kernel_size=(3,3))
    # [19,19,1024]==>[10,10,1024]
    x = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
    
    # 12th block
    # 叠加残差边[10,10,1024]
    x = layers.Add()([residual, x])
    
    # 13th block
    # [10,10,1024]==>[10,10,1536]
    x = layers.SeparableConv2D(1536, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 14th block
    # [10,10,1536]==>[10,10,2048]
    x = layers.SeparableConv2D(2048, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 15th block
    # [10,10,2048]==>[None,2048]
    x = layers.GlobalAveragePooling2D()(x)
 
    # [None,2048]==>[None,classes]
    outputs = layers.Dense(classes)(x)  # logits层不做softmax
 
    # 构建模型
    model = Model(inputs, outputs)
 
    return model
 
 
#（6）接收网络模型
if __name__ == '__main__':
    model = xception(input_shape=[299,299,3], classes=1000)
    model.summary()  # 查看网络模型结构