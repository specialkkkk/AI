import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])     #(10, )
# y = ??

#  Rnn에 쓸 수 있는 데이터로 만든다
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]]) 
y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)  #  (7, 3), (7,)

x = x.reshape(7, 3, 1)             #데이터나 순서가 바뀌면 reshape불가
print(x.shape)   # (7, 3, 1)     7개의 데이터와 /   3개씩 자른것을  / 한개씩 연산했다
                                 # =>  [[[1],[2],[3]], . . . ]

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(3, 1)))              #1은 feature =특성
                        # (N, 3, 1) -> ([batch, timesteps, feature) = 타임스텝만큼 짜르고 피쳐만큼 일을 시킨다★
model.add(Dense(35, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


# 64 * (64+ 1+ 1) = 4224
# units * (feature + units + bias) = parms(파라미터)

