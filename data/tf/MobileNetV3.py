import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Activation
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_custom_objects
def Bneck(x, kernel_size, input_size, expand_size, output_size, activation,stride,use_se):

        out = keras.layers.Conv2D(filters=expand_size,kernel_size=1,strides=1,use_bias=False)(x)
        #print(out.shape)
        out = keras.layers.BatchNormalization(axis=1)(out)
        out = activation(out)
        #print(out.shape)
        out = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=stride,padding='same')(out)
        out = keras.layers.BatchNormalization()(out)
        #print(out.shape)
        if use_se:
            out = SE_block(out)
        out = keras.layers.Conv2D(filters=output_size, kernel_size=1,strides=1,padding='same')(out)
        out = keras.layers.BatchNormalization()(out)
        if stride==1 and input_size == output_size:

            short_cut = keras.layers.Conv2D(filters=output_size,kernel_size=1,strides=1,padding='same')(x)
            out=keras.layers.Add()([out,short_cut])
        return out

def SE_block(x,reduction=4):
 channel_axis = 1 if  keras.backend.image_data_format() == "channels_first" else -1
 filters = x._shape_val[channel_axis]
 out = keras.layers.GlobalAveragePooling2D()(x)
 out = keras.layers.Dense(int(filters/reduction),activation='relu')(out)
 out = keras.layers.Dense(filters,activation='hard_sigmoid')(out)
 out = keras.layers.Reshape((1,1,-1))(out)
 out = keras.layers.multiply([x,out])
 return out
def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6


get_custom_objects().update({'custom_activation': Activation(Hswish)})

def MobileNetv3_large(num_classes=1000,input_shape=(224,224,3)):
    x = Input(shape=input_shape)
    out = keras.layers.Conv2D(16,3,strides=2,padding='same',use_bias=False)(x)
    out = keras.layers.BatchNormalization(axis=-1)(out)
    out = Activation(Hswish)(out)
    out = Bneck(out,3,16,16,16,keras.layers.ReLU(),1,False)
    out = Bneck(out,3,16,64,24,keras.layers.ReLU(),2,False)
    out = Bneck(out,3,24,72,24,keras.layers.ReLU(),1,False)
    out = Bneck(out,5,24,72,40,keras.layers.ReLU(),2,True)
    out = Bneck(out,5,40,120,40,keras.layers.ReLU(),1,True)
    out = Bneck(out,5,40,120,40,keras.layers.ReLU(),1,True)
    out = Bneck(out,3,40,240,80,Activation(Hswish),2,False)
    out = Bneck(out,3,80,200,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,184,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,184,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,480,112,Activation(Hswish),1,True)
    out = Bneck(out,3,112,672,112,Activation(Hswish),1,True)
    out = Bneck(out,5,112,672,160,Activation(Hswish),1,True)
    out = Bneck(out,5,160,672,160,Activation(Hswish),2,True)
    out = Bneck(out,5,160,960,160,Activation(Hswish),1,True)
    out = keras.layers.Conv2D(filters=960,kernel_size=1)(out)
    #print(out.shape)
    out = keras.layers.BatchNormalization()(out)
    out = Activation(Hswish)(out)
    #print(out.shape)
    out = keras.layers.AveragePooling2D(pool_size=(7, 7))(out)
    #out = keras.layers.Conv2D(1280,kernel_size=1,use_bias=False,name='out_conv')(out)
    #print(out.shape)
    out = keras.layers.Conv2D(filters=1280,kernel_size=1,strides=1)(out)
    out = Activation(Hswish)(out)
    out = keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1)(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Softmax()(out)
    model = Model(inputs=x,outputs=out)
    model.summary()
    keras.utils.plot_model(model, 'small_resnet_model.png', show_shapes=True)
    return model
model=MobileNetv3_large(num_classes=1000,input_shape=(224,224,3))