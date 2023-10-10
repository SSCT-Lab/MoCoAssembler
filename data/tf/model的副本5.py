#-- coding: utf-8 --
import tensorflow as tf
from backbone import *
from detectnet import *
from fpn import *

def create_model(num_classes):
    inputs = tf.keras.Input(shape=(None,None,3))

    c3, c4, c5 = res_net50(inputs)
    #c3, c4, c5 = ResNet2D50(inputs)
    print(f'c3.shape, c4.shape, c5.shape:{c3.shape},{c4.shape},{c5.shape}')  # (2, 28, 28, 256) (2, 14, 14, 256) (2, 7, 7, 256)
    list = fpn(c3, c4, c5)
    #(2, 28, 28, 256),(2, 14, 14, 256),(2, 7, 7, 256),(2, 4, 4, 256),(2, 2, 2, 256)
    print(f'p3~p7.shape:{list[0].shape},{list[1].shape},{list[2].shape},{list[3].shape},{list[4].shape}')
    #ouput.shap :(2, 28, 28, 9 * n_classes),(2, 14, 14, 720),(2, 7, 7, 720),(2, 4, 4, 720),(2, 2, 2, 720)
    res1, res2 = detectnet(list,num_classes)
    return tf.keras.Model(inputs=inputs,outputs=[res1, res2])
    pass

if __name__ == '__main__':
    input =tf.constant(np.random.rand(2,224,224,3),dtype=tf.float32)
    model = create_model(20)
    res1,res2 = model(input)
    print(len(model.variables))
    # for v in model.variables:
    #     print(v.name)
    #     pass
    #print(model.variables)
    print(res1.shape)
    print(res2.shape)
    # (2, 9441, 4)
    # (2, 9441, 20)
    pass