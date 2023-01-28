from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() #이미 트레인,테스트 분리가 되어있다.

# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)    # (10000, 28, 28)  (10000,)

print(x_train[1000])
print(y_train[1000])  # 5

import matplotlib.pyplot as plt
plt.imshow(x_train[10], 'gray')
plt.show()



print(x_train.shape, y_train.shape)       #(60000, 784) (60000,)
print(x_test.shape, y_test.shape)       #(10000, 784) (10000,
 


x_train = x_train/255.
x_test = x_test/255.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, LSTM

path = './_save/'

#2. 모델
model = Sequential()
model.add(LSTM(units=64, input_shape=(28,28 ), 
                activation='relu'))        
model.add(Dense(64,activation='relu'))          
model.add(Dense(80, activation='relu'))     
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.55))              
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))              
model.add(Dense(10, activation='softmax'))

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
                    filepath= filepath + 'k36_02_' + date + filename)


model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=1250,
          validation_split=0.2,callbacks=[earlyStopping,mcp])


model.save(path + "keras36_2_fashion_save_mode.h5")


#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])




# loss :  0.31241998076438904
# acc :  0.9050999879837036
#--------------------------------dnn
# loss :  0.4625706076622009
# acc :  0.8519999980926514
