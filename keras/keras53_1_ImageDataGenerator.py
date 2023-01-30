import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#x와 y를 모아서

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)  # test데이터는 rescale만 한다.

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/', #  . = 현재의 
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    #Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/', #  . = 현재의 
    target_size=(200,200),
    batch_size=10000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)



print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x00000165AD584F40>

# from sklearn.datasets import load_boston
# datasets = load_boston
# print(datasets)

print(xy_train[0])  # xy_train의 0번째를 보여줘
#batch_size를 6개 주니까 y가 6개 나옴   (array([0., 0., 0., 1., 0., 1.])
print(xy_train[0][0])
print(xy_train[0][0].shape)
#(5, 200, 200, 1)  = batch_size를 5개 줬을때 맨앞이 5나옴
print(xy_train[0][1].shape)

print(type(xy_train))  #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>




        


