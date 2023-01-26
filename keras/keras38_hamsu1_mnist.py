#cnn을 함수로형으로 !!

import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
path = './_save/'
(x_train, y_train), (x_test, y_test) = mnist.load_data() #이미 트레인,테스트 분리가 되어있다. #데이터가 이미 분리되어있기때문에 스플릿 할 필요 x

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28)  (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)      # cnn에 넣을 수있는 4차원 데이터로 바뀜
print(x_test.shape, y_test.shape)     #(60000, 28, 28, 1) (60000,)
  #(10000, 28, 28, 1) (10000,

print(np.unique(y_train , return_counts=True))  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], int 64)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input


#2. 모델
input1 = Input(shape=(28,28,1))
dense1 = Conv2D(filters=128, kernel_size=(2,2), input_shape=(28, 28, 1), activation='relu')(input1)         # (27,27,128)
dense2 = Conv2D(filters=64, kernel_size=(2,2), activation='relu')(dense1)     # (26,26,64)    
dense3 = Conv2D(filters=64, kernel_size=(2,2), activation='relu')(dense2)    # (25,25,64)
dense4 = Flatten()(dense3)
dense5 = Dense(32, activation='relu')(dense4)             # input_shape = (60000,40000)   =>  (40000,)   # (6만,4만)이 인풋이야 (batch_size, input_dim)
dense6 = Dense(32, activation='relu')(dense5)   
dense7 = Dense(32, activation='relu')(dense6) 
output1 = Dense(10, activation='softmax')(dense7)
model = Model(inputs=input1, outputs=output1)

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
                    filepath= filepath + 'k34_01_' + date + filename)


model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32,
          validation_split=0.25,callbacks=[earlyStopping,mcp])


model.save(path + "keras34_1_mnist_save_mode.h5")

#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])     







#es,mcp,val 추가했음
#acc :  0.9663000106811523