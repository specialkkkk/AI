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


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)      # cnn에 넣을 수있는 4차원 데이터로 바뀜
print(x_test.shape, y_test.shape)     #(60000, 28, 28, 1) (60000,)
  #(10000, 28, 28, 1) (10000,

print(np.unique(y_train , return_counts=True)) 



from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input

path = './_save/'

#2. 모델

input1 = Input(shape=(28,28,1))
dense1 = Conv2D(filters=128, kernel_size=(2,2), activation='relu',
                padding='same')(input1)           
dense2 = MaxPooling2D()(dense1)
dense3 = Conv2D(filters=64, kernel_size=(2,2),activation='relu', padding='same')(dense2)
dense4 = Dropout(0.5)(dense3) 
dense5 = Conv2D(filters=64, kernel_size=(2,2),activation='relu',  padding='same')(dense4)        
dense6 = Flatten()(dense5)
dense7 = Dense(32, activation='relu')(dense6)
dense8 = Dense(16, activation='relu')(dense7)
dense9 = Dense(8, activation='relu')(dense8)                                                  
output1 = Dense(10, activation='softmax')(dense9)
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
                    filepath= filepath + 'k35_02_' + date + filename)


model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=1250,
          validation_split=0.2,callbacks=[earlyStopping,mcp])


model.save(path + "keras35_2_fashion_save_mode.h5")


#4. 평가,예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])




# loss :  0.31241998076438904
# acc :  0.9050999879837036

