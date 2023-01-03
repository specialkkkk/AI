import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1.데이터
x = np.array(range(10))   # (10,)  

print(x.shape)          

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])


y = y.T
print(y.shape)           # (10,3)

  
# 2.모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(20))
model.add(Dense(88))
model.add(Dense(89))
model.add(Dense(99))
model.add(Dense(102))
model.add(Dense(105))
model.add(Dense(88))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('[6]의 예측값 : ', result) 



