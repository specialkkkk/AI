import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
path = './_save/'
(x_train, y_train), (x_test, y_test) = mnist.load_data() #이미 트레인,테스트 분리가 되어있다. #데이터가 이미 분리되어있기때문에 스플릿 할 필요 x

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28)  (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

#dnn할때 reshape   (전에 하던것은 cnn)
x_train = x_train.reshape(60000, 28*28)   
x_test = x_test.reshape(10000, 28*28)      

print(x_train.shape, y_train.shape)      # (60000, 784) (60000,)     784개의 컬럼으로 바꿨다. 
print(x_test.shape, y_test.shape)     #(10000, 784) (10000,)

x_train = x_train/255.
x_test = x_test/255.

print(np.unique(y_train , return_counts=True))  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], int 64)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout





#2. 모델
model = Sequential()          #  28*28 = 784
model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))




model.summary()




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
                    filepath= filepath + 'k36_01_' + date + filename)


model.fit(x_train, y_train, epochs=300, verbose=1, batch_size=32,
          validation_split=0.25,callbacks=[earlyStopping,mcp])


model.save(path + "keras36_1_mnist_save_mode.h5")

#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])     






#--------------------------------cnn
#es,mcp,val 추가했음
#acc :  0.9663000106811523
#첫레이어 커널사이즈 3,3    +  두번째까지 패딩 = acc : 0.9782999753952026
#맥스풀 추가 =  acc :  0.982699990272522
#세번째까지 패딩 =  acc :  0.9836999773979187


#---------------------------------dnn
# loss :  0.16145645081996918
# acc :  0.9635999798774719

