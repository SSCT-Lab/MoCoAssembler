import copy
import numpy as np
import tensorflow as tf


def lenet(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=6, kernel_size=6, strides=1, activation="relu", padding="same")(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=200, activation="relu")(x)
    x = tf.keras.layers.Dense(units=10, activation="softmax")(x)
    output_tensor = tf.keras.layers.Flatten()(x)
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model




def go():
    with tf.device('/GPU:0'):
       tf_input = tf.random.normal([1, 28, 28, 1])
       tf_model = lenet(tf_input.shape[1:])
       tf_output = tf_model(tf_input)
       return tf_output




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
    model_g = lenet(inp.shape[1:])
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

