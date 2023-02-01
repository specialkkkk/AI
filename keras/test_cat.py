import numpy as np
import pandas as pd 
import random
import os

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import os










train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range= 5,
    # zoom_range=1.2,
    # shear_range=0.7, 
    #  fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'd:./_data/catdog/train/', 
    target_size=(200, 200),
    batch_size=1000,
    class_mode='binary',
    # class_mode='categorical', # 원핫
    color_mode='rgb',
    shuffle=True,
     # Found 160 images belonging to 2 classes.
)


xy_test = train_datagen.flow_from_directory(
    'd:./_data/catdog/test/', 
    target_size=(200, 200),
    batch_size=1000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
     # Found 120 images belonging to 2 classes.
)

y = train_datagen.flow_from_directory(
    'd:./_data/catdog/predict/', 
    target_size=(200, 200),
    batch_size=1000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
     # Found 120 images belonging to 2 classes.
)

# print(y[0][0].shape)
# print(xy_test[0][0].shape)
# # train_datagen.flow_from_directory()
# xy_test = train_datagen.flow_from_directory(
#     'd:./_data/catdog/train/', 
#     target_size=(200, 200),
#     batch_size=100000,
#     # class_mode='categorical', # 원핫
#     class_mode='binary',
#     color_mode='grayscale',
#     shuffle=True,
#      # Found 120 images belonging to 2 classes.
# )

# print(xy_train) # Found 25000 images belonging to 1 classes.
# Found 12500 images belonging to 1 classes.
# <keras.preprocessing.image.DirectoryIterator object at 0x0000019861EA29A0>

# print(xy_train[0][0].shape)  # (25000, 200, 200, 1)
# print(xy_train[0][1].shape)  # (25000,)



np.save('d:./_data/catdog/x_train.npy', arr = xy_train[0][0])
np.save('d:./_data/catdog/y_train.npy', arr = xy_train[0][1])
np.save('d:./_data/catdog/x_test.npy', arr = xy_test[0][0])
np.save('d:./_data/catdog/y_test.npy', arr = xy_test[0][1])
np.save('d:./_data/catdog/y4_predict.npy', arr = y[0][0])


# # batch_size 를 크게 잡았을때 데이터의 개수를 확인할수 있다.
# print (type(xy_train[0])) # <class 'tuple'>
# print(type(xy_train[0][0]))# <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))# <class 'numpy.ndarray'>