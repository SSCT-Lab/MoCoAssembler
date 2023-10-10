import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization,MaxPool2D,Flatten,Dropout,Dense
from tensorflow.keras import Model
import os
import numpy as np

cifar10=tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

class LeNet5(Model):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.c1=Conv2D(filters=6,kernel_size=(5,5),strides=1,padding='valid')
        self.a1=Activation('sigmoid')
        self.p1=MaxPool2D(pool_size=(2,2),strides=2,padding='valid')

        self.c2=Conv2D(filters=16,kernel_size=(5,5),strides=1,padding='valid')
        self.a2=Activation('sigmoid')
        self.p2=MaxPool2D(pool_size=(2,2),strides=2,padding='valid')

        self.flatten3=Flatten()

        self.d4=Dense(120,activation='sigmoid')

        self.d5=Dense(84,activation='sigmoid')

        self.d6=Dense(10,activation='softmax')

    def call(self,x):
        x=self.c1(x)
        x=self.a1(x)
        x=self.p1(x)

        x=self.c2(x)
        x=self.a2(x)
        x=self.p2(x)

        x=self.flatten3(x)
        x=self.d4(x)
        x=self.d5(x)
        y=self.d6(x)
        return y

model=LeNet5()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

history=model.fit(x_train,y_train,batch_size=32,epochs=10,
                  validation_data=(x_test,y_test),
                  validation_freq=1)

model.summary()


###################  绘制训练集和验证集的损失和精度曲线 ##################
acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()