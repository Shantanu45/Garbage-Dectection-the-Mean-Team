# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:41:52 2019

@author: Shantanu
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

garbage = np.load("D:/meanTeam/Garbage/model/garbage.npy")
non_garbage = np.load("D:/meanTeam/Garbage/model/nongarbage.npy")

label = np.append(np.ones(garbage.shape[0]),np.zeros(non_garbage.shape[0]))
data = np.concatenate((garbage, non_garbage), axis=0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42 )
#
#from tensorflow.data_preprocessing import ImagePreprocessing
#from tflearn.data_augmentation import ImageAugmentation
#
#img_prep = ImagePreprocessing()
#img_prep.add_featurewise_zero_center()
#img_prep.add_featurewise_stdnorm()
MEAN = 0
STDDIV = 0
def scalar(arr, img):
    global MEAN
    global STDDIV 
    MEAN = np.mean(img)
    STDDIV = np.std(img)
    sub = np.subtract(arr, MEAN)
    div = np.divide(sub, max(STDDIV, 1/math.sqrt(img.shape[0]*img.shape[1]*3*img.shape[3])))
    return div

#from tensorflow.keras import backend
#
#X_train_norm = tf.image.per_image_standardization(tf.image.per_image_standardization(X_train))
#X_train_array = backend.eval(X_train_norm)
#X_test_norm = tf.image.per_image_standardization(tf.image.per_image_standardization(X_test))
#X_test_array = backend.eval(X_test_norm)
    
X_train_array = scalar(X_train, X_train)
X_test_array = scalar(X_test, X_train)

model = keras.models.Sequential([
        keras.layers.Conv2D(64, 11, activation="relu",padding="same", input_shape=[227, 227, 3], strides=4),
        keras.layers.MaxPool2D(3, strides=2),
        keras.layers.Conv2D(256, 5, activation="relu", padding="same"),
        keras.layers.MaxPool2D(3, strides=2),
        keras.layers.Conv2D(384, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(384, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.MaxPool2D(3, strides=2),
        keras.layers.Conv2D(512, 6, activation="relu", padding="same", strides=2),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(256, 1, activation="relu", padding="same"),      
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
#        keras.layers.Dense(150, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
        ])

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
#adam

history = model.fit(X_train_array, y_train, epochs=20, validation_split=0.1)
#
##X_test_norm = tf.image.per_image_standardization(tf.image.per_image_standardization(X_test))
##X_test_array = backend.eval(X_test_norm)
#
model.evaluate(X_test_array, y_test)

model.save("GarbageModel_acc-0.8672.h5")
np.save("scalar", np.array([MEAN, STDDIV]))
np.save("sizeImg", X_train.shape)