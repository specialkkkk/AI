from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)   # 50000, 32, 32, 3  (50000,)          y=5만개 있다는 뜻
print(x_test.shape, y_test.shape)    # 10000, 32, 32, 3    (10000, 1)

print(np.unique(y_train , return_counts=True))  # (array ([0~9]))   # (10,)

x_train = x_train.reshape(50000, 32*32*3)   
x_test = x_test.reshape(10000, 32*32*3)

x_train = x_train/255.
x_test = x_test/255.
#  /255.  = mimax scale     #원핫인코더와 다른것

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
path = './_save/'

#2. 모델
model = Sequential()
model.add(Dense(284, input_shape=(32*32*3, ), activation='relu'))
model.add(Dropout(0.5))      
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))      
model.add(Dense(86, activation='relu'))        
model.add(Dense(48, activation='relu'))  
model.add(Dense(32, activation='relu'))   
model.add(Dense(16, activation='relu'))                                                           
model.add(Dense(10, activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
                    filepath= filepath + 'k36_03_' + date + filename)


model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=250,
          validation_split=0.2, callbacks=[earlyStopping,mcp])


model.save(path + "keras36_3_cifar10_save_mode.h5")


#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


# 로스는 0 에 가까울수록 좋다.
# acc는 1에 가까울수록 좋다.


# padding + Maxpooling  loss :  1.089743733406067
# acc :  0.6186000108718872


#dropout 하나 빼기           loss :  1.047391653060913
# acc :  0.6521000266075134


#dropout 전부 빼기 
# loss :  2.3026392459869385
# acc :  0.10000000149011612


# ------------------------dnn
# loss :  2.30196475982666
# acc :  0.1006999984383583

# acc :  0.15549999475479126
