from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import pandas as pd
import numpy as np

#1. 데이터
path = './_data/bike/'
train_data = pd.read_csv(path + 'train.csv', index_col = 0)         # index_col = 0 → date_t 열 데이터로 취급 X
test_data = pd.read_csv(path + 'test.csv', index_col = 0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

# print(train_data.shape)          # (10886, 11) 
# print(test_data.shape)           # (6493, 8)
# print(train_data.columns)   
# print(train_data.info())         # Missing Attribute Values: 결측치 - 데이터에 값이 없는 것
# print(train_data.describe())   # 평균, 표준편차, 최대값 등

#  shape 맞추기 (열 제거) #
train_data = train_data.drop(['casual', 'registered'], axis = 1)

#  x,y 분리 #
x = train_data.drop(['count'], axis=1)                              # y 값(count 열) 분리, axis = 1 → 열에 대해 동작
y = train_data['count']                                             # y 값(count 열)만 추출

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=123
)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_data)








# #2. 모델 구성
# model = Sequential()
# model.add(Dense(5, input_dim = 8))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(1))


#2. 모델구성(함수형)
input1 = Input(shape=(8,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()



#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=5, restore_best_weights=True,
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
                    filepath= filepath + 'k31_05_' + date + filename)

hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2, callbacks=[earlyStopping,mcp],
          verbose=1)



#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) 
print('loss: ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE)

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)



# 제출
y_submit = model.predict(test_data)
submission['count'] = y_submit
submission.to_csv(path + 'samplesubmission_0109 bike.csv')

print("=======================")
print(hist) #<keras.callbacks.History object at 0x0000024787770D90>
print("=======================")
print(hist.history)
print("=======================")
print(hist.history['loss'])
print("=======================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('bike loss')
plt.legend()

#plt.legend(loc='upper right')   = 오른쪽 위로 위치 지정

plt.show()


# RMSE:  149.2192691427927
# R2:  0.31460662588718424
