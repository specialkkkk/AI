#실습 LSTM 모델구성
# x_predict = np.array([7,8,9,10])      # => 결과는 11이 나와야 한다!!!!!!!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout,LSTM

a = np.array(range(1, 11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []                                             # 출력할거 담을 빈 리스트
    for i in range(len(dataset) - timesteps + 1):  # 범위-타입스텝+1 만큼 반복    =>  (0, 1, 2) 가 i로 들어간다 = 0 
            subset = dataset[i : (i + timesteps)]
            aaa.append(subset)
    return np.array(aaa)
    
bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)    # (6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)
print(x.shape, y.shape)  # (6, 4) (6, )




x = x.reshape(6, 4, 1)             
print(x.shape)   # (6, 4, 1)    

    



#2. 모델구성
model = Sequential()   # (N, 4, 1) -> ([batch, timesteps, feature) = 타임스텝만큼 짜르고 피쳐만큼 일을 시킨다★
model.add(LSTM(units=64, input_shape=(4, 1), activation='relu'))  
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=350)


#4. 평가 예측
loss = model.evaluate(x,y)
print(loss)
x_pred = np.array([7, 8, 9, 10]).reshape(1, 4, 1)    #1차원을 3차원으로 바꾼다(reshape) = 가로가 늘어난 것 뿐
                                                    
result = model.predict(x_pred)
print('[7,8,9,10]의 결과 : ' , result )

# [7,8,9,10]의 결과 :  [[11.12119]]            11을 만들자!













'''
timesteps = 3
[[1 2 3]
 [2 3 4]
 [3 4 5]]
 
 

timesteps = 4

[[1 2 3 4]
 [2 3 4 5]]
 '''