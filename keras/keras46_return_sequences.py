# x_predict = np.array([50,60,70])     # 과제 80!!!


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout,LSTM
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],
              [4,5,6],[5,6,7],[6,7,8],
              [7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])



print(x.shape, y.shape)  #  (13, 3), (13,)

x = x.reshape(13, 3, 1)             
print(x.shape)   # (13, 3, 1)     # (13개가 있다, 3개짜리가, 1개씩 )  

    



#2. 모델구성
model = Sequential()   # (N, 13, 1) -> ([batch, timesteps, feature) = 타임스텝만큼 짜르고 피쳐만큼 일을 시킨다★
model.add(LSTM(units=64, input_shape=(3, 1), return_sequences=True))                # LSTM 두번쓰면에러 => return_sequences를 쓴다 = 시퀀스를 돌려주겠다 
model.add(LSTM(30, activation='relu'))
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
x_pred = np.array([50, 60, 70]).reshape(1, 3, 1)    #1차원을 3차원으로 바꾼다(reshape) = 가로가 늘어난 것 뿐
                                                    #  [50,60,70]   reshape(한개있다, 3개짜리가, 1개씩)  
result = model.predict(x_pred)
print('[50,60,70]의 결과 : ' , result )

# [50,60,70]의 결과 :  [[81.26452]]            80을 만들자!
