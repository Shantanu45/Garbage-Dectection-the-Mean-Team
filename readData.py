# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:59:03 2019

@author: Shantanu
"""

import os
#import cv2
import numpy as np
from PIL import Image

# Get the list of all files in directory tree at given path
listOfFiles = list()
dirpath = "../dataset/spotgarbage-GINI/spotgarbage/non-garbage-queried-images/"
for root, directories, filenames in os.walk(dirpath):
    listOfFiles += [os.path.join(root, file) for file in filenames]
    
images = []
    
for i in listOfFiles:
    input_image = Image.open(i)
    images.append(input_image)
    


def resizeForFCN(image,size):
    w,h = image.size
    if w<h:
        return image.resize((int(227*size),int((227*h*size)/w))) #227x227 is input for regular CNN
    else:
        return image.resize((int((227*w*size)/h),int(227*size)))
    
a = []
for j in range(len(images)):
    test = resizeForFCN(images[j], 1)
    old_size = test.size

    new_size = (227, 227)
    new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
    new_im.paste(test, (int((new_size[0]-old_size[0])/2),
                      int((new_size[1]-old_size[1])/2)))
    
    a.append(np.array(new_im))
    
nparray = np.array(a)
np.save("nongarbage", nparray)
#arr2 = np.append(arr, [1])
#print(arr2[-2])
    
#cv2.imshow("image", img)
#cv2.waitkey(0)
#cv2.destroyAllWindows()
