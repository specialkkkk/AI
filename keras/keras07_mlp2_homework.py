import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])

y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape) # (2, 10)
print(y.shape) # (10,)

x = x.T
print(x.shape) # (10,2)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 1.4, 0]])
print('[10, 1.4, 0]의 예측값 : ', result)

'''
X가 3덩어리 = input_dim은 3이 된다.
predict의 값도 바꿔준다. (10, 1.4, 0) 3가지를 써줌.
'''