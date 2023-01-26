from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

path = './_save/'

# 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

x_train = x_train/255.
x_test = x_test/255.


'''
# 모델구성
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))
'''
input1 = Input(shape=(32,32,3))
dense1 = Conv2D(32, kernel_size=(3, 3), activation='relu',
                 padding='same')(input1)

dense2 = MaxPooling2D(pool_size=(2, 2))(dense1)
dense3 = Conv2D(64, kernel_size=(3, 3), activation='relu')(dense2)
dense4 = MaxPooling2D(pool_size=(2, 2))(dense3)
dense5 = Flatten()(dense4)
dense6 = Dense(512, activation='relu')(dense5)
dense7 = Dropout(0.5)(dense6)
output1 = Dense(100, activation='softmax')(dense7)
model = Model(inputs=input1, outputs=output1)






# 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=20, restore_best_weights=True,
                              verbose=1)

import datetime
date = datetime.datetime.now()
print(date)    # 2023-01-12 14:58:02.348691
print(type(date))     # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   #0112_1457
print(date)    # 0112_1502
print(type(date))

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #0037-0.0048.hdf5 




mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath= path  + 'MCP/keras30_ModelCheckPoint3.hdf5')        
                    filepath= filepath + 'k38_04_' + date + filename)


model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=1250,
          validation_split=0.25,callbacks=[earlyStopping,mcp])




model.save(path + "keras35_4_cifar100_save_mode.h5")



#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])




# loss :  2.3853020668029785
# acc :  0.3991999924182892