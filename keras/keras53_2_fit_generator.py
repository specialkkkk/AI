import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

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
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    #Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/', #  . = 현재의 
    target_size=(100,100),
    batch_size=10000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
     #Found 120 images belonging to 2 classes.
)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100,100,1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit_generator(xy_train, steps_per_epoch=16,  #epoch당 배치 몇번?
                    epochs=100,validation_data=xy_test,
                    validation_steps=4, )  

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])   # loss[100]하면 100번째가 나오지만  =  [-1]하면 뒤에서 첫번째가 나온다
print('val_loss : ', val_loss[-1]) 
print('accuracy : ', accuracy[-1])
print('val_acc : ',val_acc[-1])




        


