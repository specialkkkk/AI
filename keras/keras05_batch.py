import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=50, batch_size=5)

#4. 평가, 예측
result = model.predict([6])
print('6의 결과 : ', result)

"""
batch_size=1 6/6  
batch_size=2 3/3
batch_size=3 2/2
batch_size=4 2/2
batch_size=5 2/2
batch_size=6 1/1
"""
