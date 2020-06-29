# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 21:19:14 2020

@author: vasan
"""
# import libraries 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#load MNIST data - split dataset 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

numb_class = 10          # number of classes 0,1,..9
input_shape = (28, 28, 1)  #grayscale image of size 28*28*1

# show image  
plt.imshow(x_train[123])
print("The displayed digit is", y_train[123])

# scale the images from 0 to 1
x_train = x_train.astype("float32")/255 # grayscale values range from 0-255
x_test = x_test.astype("float32")/255

# reshape x_train and x_test matrix
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("Shape of x train matrix", x_train.shape)
print("Number of training samples", x_train.shape[0])
print("Number of test samples", x_test.shape[0])

# converting to categorical matrics
y_train = keras.utils.to_categorical(y_train, numb_class)
y_test = keras.utils.to_categorical(y_test, numb_class)

# build the model
model = keras.Sequential(
        [
                keras.Input(shape = input_shape),
                layers.Conv2D(32, kernel_size = (3,3), activation = 'relu'),
                layers.MaxPooling2D(pool_size=(2,2)), 
                layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
                layers.MaxPooling2D(pool_size = (2,2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(numb_class, activation = 'softmax'),
                              
        ])

model.summary()

batch_size = 128
epochs = 15

# compile and fit model
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'] )
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
    
# evaluate model 
score = model.evaluate(x_test, y_test, verbose = 0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])



