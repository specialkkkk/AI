from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)                                 # (50000, 32, 32, 3) (50000, 1)
                                                                    # 데이터 내용이나 순서의 영향을 받지 않는다. reshape
print(x_test.shape, y_test.shape)                                   # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))                       

                                                                    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model = Sequential()
model.add(Conv2D(filters=512, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(256, (3,3)))
model.add(Conv2D(128, (3,3)))
model.add(Conv2D(64, (3,3)))
model.add(Conv2D(32, (3,3)))
model.add(Flatten())
model.add(Dense(1250, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='linear'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='softmax'))                                     # array가 0부터 99까지이니 100개

#3. 컴파일, 훈련    
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', 
                            #   restore_best_weights=False,
                              verbose=1)

import datetime                                                                 # 데이터 타임 임포트해서 명시해준다.
date = datetime.datetime.now()                                                  # 현재 날짜와 시간이 date로 반환된다.

print(date)                                                                     # 2023-01-12 14:57:54.345489
print(type(date))                                                               # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")                                               # date를 str(문자형)으로 바꾼다.
                                                                                # 0112_1457
print(date)
print(type(date))                                                               # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                                    # -는 연산 -가 아니라 글자이다. str형이기 때문에...
                                                                                # 0037-0.0048.hdf5

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath=path + 'MCP/keras31_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k34_03' + date + '_' + filename
)
                                                                                       
model.fit(x_train, y_train, epochs=1200, batch_size=1025,
          validation_split=0.25,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
# model.save(path + "keras31_ModelCheckPoint3_save_model.h5")

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])                                                     # loss 값 반환
print('acc:', results[1])                                                      # acc 값 반환

# es, mcp 적용 / val 적용



