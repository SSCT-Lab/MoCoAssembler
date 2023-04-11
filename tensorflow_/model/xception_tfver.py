import tensorflow as tf
from tensorflow.python.keras import Input, layers, models
from tensorflow.python.layers.normalization import BatchNormalization


def xception(input_shape, classes):
    input_tensor = Input(shape=input_shape)

    # 1st block
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu", use_bias=False)(input_tensor)
    x = BatchNormalization()(x)

    # 2nd block
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu", use_bias=False)(x)
    x = BatchNormalization()(x)

    # 3rd block
    # 残差边
    residual = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # 4th block
    x = layers.SeparableConv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # 5th block
    x = layers.SeparableConv2D(filters=128, kernel_size=3, strides=1, padding='same', activation="relu",
                               use_bias=False)(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # 6th block
    x = res_block(x, filters=256)

    # 7th block
    x = res_block(x, filters=728)

    # 8th block
    x = middle_flow(x, filters=728)

    # 9th block
    residual = layers.Conv2D(filters=1024, kernel_size=1, strides=2, use_bias=False, padding='same')(x)
    residual = BatchNormalization()(residual)  # 批标准化

    # 10th block
    x = layers.SeparableConv2D(filters=728, kernel_size=3, strides=1, padding='same', activation="relu",
                               use_bias=False)(x)

    # 11th block
    x = layers.SeparableConv2D(filters=1024, kernel_size=3, strides=1, padding='same', activation="relu",
                               use_bias=False)(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # 12th block
    x = layers.Add()([residual, x])

    # 13th block
    x = layers.SeparableConv2D(1536, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 14th block
    x = layers.SeparableConv2D(2048, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 15th block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes)(x)

    output_tensor = x

    # 构建模型
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model

 
#（3）一个残差单元
def res_block(input_tensor, filters):
 
    # ① 残差边
    residual = layers.Conv2D(filters=filters, kernel_size=1, strides=2)(input_tensor)  # 使输入和输出的size相同
 
    residual = BatchNormalization()(residual)  # 批标准化
 
    # ② 卷积块
    x = layers.SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu",
                               use_bias=False)(input_tensor)
    x = layers.SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu",
                               use_bias=False)(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
 
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
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu",
                                   use_bias=False)(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu",
                                   use_bias=False)(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation="relu",
                                   use_bias=False)(x)
        # 叠加残差边
        x = layers.Add()([residual, x])
 
    return x
 

if __name__ == '__main__':
    model = xception(input_shape=(299, 299, 3), classes=1000)
    model.summary()
