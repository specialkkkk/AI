import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1, 17))          # 1~ 16
y = np.array(range(1, 17))
# [실습] 슬라이싱으로 잘라라!!

x_train = x[:11]
x_test = x[11:]
y_train = y[:11]
y_test = y[11:]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

x_validation = np.array([12,13,14,15,16])
y_validation = np.array([12,13,14,15,16])



# x_train = np.array(range(1, 11))  ??깃헙
# y_train = np.array(range(1, 11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_validation = np.array([14,15,16])
# y_validation = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation))



#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)


