from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)       # 판다스 .describe()    /     .info()

'''
Y = 컬럼4개 input_dim=4
 
- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm
'''
print(datasets.feature_names)          # 판다스 .columns


x = datasets.data
y = datasets['target']
print(x)   
print(y)   
print(x.shape, y.shape)    # (150,4)  ,  (150,)
                           # x = 150짜리 1개 = 1
                           
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.9,
                                                    stratify=y)
print(y_train)
print(y_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))             # 3 = y의 클래스의 개수


# #2. 모델구성(함수형)
# input1 = Input(shape=(4,))
# dense1 = Dense(50, activation='relu')(input1)
# dense2 = Dense(40, activation='relu')(dense1)
# dense3 = Dense(30, activation='relu')(dense2)
# dense4 = Dense(20, activation='relu')(dense3)
# output1 = Dense(3, activation='softmax')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


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
                    filepath= filepath + 'k31_07_' + date + filename)
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2,
          verbose=1)

#4. 평가,예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)   
print(acc)

# accuracy :  0.93 Scaler