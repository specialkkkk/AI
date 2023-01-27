import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])     #(10, )
# y = ??

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]]) 
y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)  #  (7, 3), (7,)

x = x.reshape(7, 3, 1)             #데이터나 순서가 바뀌면 reshape불가
print(x.shape)   # (7, 3, 1)     7개의 데이터와 /   3개씩 자른것을  / 한개씩 연산했다
                                 # =>  [[[1],[2],[3]], . . . ]

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(128, input_shape=(3, 1)))  #Flatten 할 필요 없음
model.add(Dense(35, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=350)

#4. 평가 예측
loss = model.evaluate(x,y)
print(loss)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)    #1차원을 3차원으로 바꾼다(reshape) = 가로가 늘어난 것 뿐
result = model.predict(y_pred)
print('[8,9,10]의 결과 : ' , result )



# [8,9,10]의 결과 :  [[10.668669]]           * 11이 나와야 한다