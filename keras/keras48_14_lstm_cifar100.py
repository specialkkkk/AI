
from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)   # 50000, 32, 32, 3  (10000,) y=5만개 있다는 뜻
print(x_test.shape, y_test.shape)    # 10000, 32, 32, 3

print(np.unique(y_train , return_counts=True))  # (array ([0~99]))   # (10,)

x_train = x_train.reshape(50000, 32,32*3)   
x_test = x_test.reshape(10000, 32,32*3) 


x_train = x_train/255.
x_test = x_test/255.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout,MaxPooling2D,LSTM

path = './_save/'

#2. 모델
model = Sequential()
model.add(LSTM(units=64, input_shape=(32,32*3), activation='relu'))        
model.add(Dense(128,activation='relu'))         
model.add(Dense(256,activation='relu'))     
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.55))              
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))              
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
                    filepath= filepath + 'k36_04_' + date + filename)


model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=1250,
          validation_split=0.25,callbacks=[earlyStopping,mcp])


model.save(path + "keras36_4_cifar100_save_mode.h5")


#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

#loss :  4.605201244354248
# acc :  0.009999999776482582





# -----------------------------dnn

# loss :  3.3501577377319336
# acc :  0.2069000005722046