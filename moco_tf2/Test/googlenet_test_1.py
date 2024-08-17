import copy
import numpy as np
import tensorflow as tf


def googlenet_test(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = inception(inputs=x, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    output_tensor = tf.keras.layers.Dense(units=1000, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def inception(inputs,ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5,pool_proj):
    x1 = tf.keras.layers.Conv2D(filters=ch1x1, kernel_size=1, activation="relu")(inputs)
    x2 = tf.keras.layers.Conv2D(filters=ch3x3red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x2 = tf.keras.layers.Conv2D(filters=ch3x3, kernel_size=3, strides=1, padding="same", activation="relu")(x2)
    x3 = tf.keras.layers.Conv2D(filters=ch5x5red, kernel_size=1, strides=1, padding="same", activation="relu")(inputs)
    x3 = tf.keras.layers.Conv2D(filters=ch5x5, kernel_size=5, strides=1, padding="same", activation="relu")(x3)
    x4 = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inputs)
    x4 = tf.keras.layers.Conv2D(filters=pool_proj, kernel_size=1, strides=1, padding="same", activation="relu")(x4)
    target_height = inputs.shape[1]
    target_width = inputs.shape[2]
    x1 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x1)
    x2 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x2)
    x3 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x3)
    x4 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (target_height, target_width)))(x4)
    outputs = tf.keras.layers.concatenate(inputs=[x1,x2,x3,x4])

    return outputs



def go():
    with tf.device('/GPU:0'):
       tf_input = tf.random.normal([1, 224, 224, 3])
       tf_model = googlenet_test(tf_input.shape[1:])
       tf_output = tf_model(tf_input)
       return tf_output


print(go())

def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    if A.shape != B.shape:
        return 9999999
    else:
        return float(np.max(np.abs(A - B)))


def train(inp, label):
    flag = True
    label = tf.convert_to_tensor(label)
    model_g = googlenet_test(inp.shape[1:])
    with tf.device('GPU'):
        with tf.GradientTape() as tape:
            output_g = model_g(inp)
            loss_g = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, output_g)
        gradients_g = tape.gradient(loss_g, model_g.trainable_variables)
        gradients_dic_g = {}
        for var, gradient in zip(model_g.trainable_variables, gradients_g):
            if gradient != None:
                gradients_dic_g.setdefault(var.name.replace('/', '.')[:-2], gradient)


    model_c = copy.deepcopy(model_g)
    with tf.device('CPU'):
        with tf.GradientTape() as tape:
            output_c = model_c(inp)
            loss_c = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(label, output_c)
        gradients_c = tape.gradient(loss_c, model_c.trainable_variables)
        gradients_dic_c = {}
        for var, gradient in zip(model_c.trainable_variables, gradients_c):
            if gradient != None:
                gradients_dic_c.setdefault(var.name.replace('/', '.')[:-2], gradient)
    if chebyshev_distance(output_c.numpy(), output_g.numpy()) > 1.0:
        flag = False
        return flag, 'Output diff too big'
    if abs(loss_c - loss_g) > 0.1:
        flag = False
        return flag, 'Loss diff too big'
    for name in gradients_dic_c.keys():
        if name in gradients_dic_g.keys():
            if chebyshev_distance(gradients_dic_c[name], gradients_dic_g[name]) > 0.1:
                flag = False
                return flag, 'Grad diff too big'
    for name in gradients_dic_g.keys():
        if name in gradients_dic_c.keys():
            if chebyshev_distance(gradients_dic_g[name], gradients_dic_c[name]) > 0.1:
                flag = False
                return flag, 'Grad diff too big'
    return flag, ''

