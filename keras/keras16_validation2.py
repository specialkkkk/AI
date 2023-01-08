import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error


# [실습] 슬라이싱으로 잘라라!
##1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
x_train = x[:10]
y_train = y[:10]
x_test = x[10:13]
y_test = y[10:13]
x_val = x[13:]
y_val = y[13:]

# x_train = np.array(range(1, 11))
# y_train = np.array(range(1, 11))
# x_test = np.array([11, 12, 13])
# y_test = np.array([11, 12, 13])
# x_validation = np.array([14, 15, 16])
# y_validation = np.array([14, 15, 16])

##2. 모델
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

##3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=80, batch_size=1,
          validation_data=(x_val, y_val))

##4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print('17의 예측값:', result)

