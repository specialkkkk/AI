from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, LSTM
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(x.shape) # (442,10)
print(y)
print(y.shape) # (442,)

print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)   #(309, 10) (133, 10)



x_train = x_train.reshape(309, 10, 1)
x_test = x_test.reshape(133, 10, 1)
print(x_train.shape, x_test.shape)

#2. 모델구성
model = Sequential()
model.add(LSTM(7, (2,1), input_shape=(10,1)))   # 2,1은 커널사이즈
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))


#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=5, restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

print("====================")
print(y_test)
print(y_predict)
print("====================")


 
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

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
plt.title('diabetes loss')
plt.legend()

#plt.legend(loc='upper right')   = 오른쪽 위로 위치 지정

plt.show()

