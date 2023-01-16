from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)   # 50000, 32, 32, 3  (10000,) y=5만개 있다는 뜻
print(x_test.shape, y_test.shape)    # 10000, 32, 32, 3

print(np.unique(y_train , return_counts=True))  # (array ([0~99]))   # (10,)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

path = './_save/'

#2. 모델
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu'))           # (31, 31, 128)
model.add(Conv2D(filters=128, kernel_size=(2,2),activation='relu'))      # (30,30,64)    
model.add(Conv2D(filters=256, kernel_size=(2,2),activation='relu'))      # (29,29,64)
model.add(Flatten())
model.add(Dense(512, activation='relu'))              
model.add(Dense(512, activation='relu'))              
model.add(Dense(100, activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = earlyStopping = EarlyStopping(monitor='val_loss' , 
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
                    filepath= filepath + 'k34_03_' + date + filename)


model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=250,
          validation_split=0.2,callbacks=[earlyStopping,mcp])


model.save(path + "keras34_3_cifar100_save_mode.h5")


#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
