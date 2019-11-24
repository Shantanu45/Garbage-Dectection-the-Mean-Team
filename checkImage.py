# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:03:20 2019

@author: Shantanu
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import math

scalar = np.load("./scalar.npy")
sizeImg = np.load("./sizeImg.npy")

MEAN = scalar[0]
STDDIV = scalar[1]
BLACK = [0,0,0]

img = "one.jpg"

checkImage = Image.open(img)


def resizeForFCN(image,size):
    w,h = image.size
    if w<h:
        return image.resize((int(227*size),int((227*h*size)/w))) #227x227 is input for regular CNN
    else:
        return image.resize((int((227*w*size)/h),int(227*size)))
    
def scalar(arr):
    global MEAN
    global STDDIV 
#    MEAN = np.mean(img)
#    STDDIV = np.std(img)
    sub = np.subtract(arr, MEAN)
    div = np.divide(sub, max(STDDIV, 1/math.sqrt(sizeImg[0]*sizeImg[1]*3*sizeImg[3])))
    return div
    
test = resizeForFCN(checkImage, 1)
old_size = test.size

new_size = (227, 227)
new_im = Image.new("RGB", new_size)
new_im.paste(test, (int((new_size[0]-old_size[0])/2),
                      int((new_size[1]-old_size[1])/2)))

img_arr = np.array(new_im)
#cv2.imshow("img", img_arr)
height, width = 227, 227

model = keras.models.load_model("GarbageModel_acc-0.8672.h5")

#from tensorflow.keras import backend

#X_norm = tf.image.per_image_standardization(img_arr)
X_array = scalar(img_arr)

X_array = np.expand_dims(X_array, axis=0)

y_pred = model.predict_classes(X_array)

if y_pred == 1:
    print("Garbage")
else:
    print("Not Grabage")