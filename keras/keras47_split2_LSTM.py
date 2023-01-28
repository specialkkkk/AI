import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout,LSTM
x_predict = np.array(range(96, 106))  # 예상 y = 100, 107
#                                        결과값은 100부터 106까지 나와야한다.(7개)
# 프레딕트= 10개  5개씩 잘라서  하나는 x고 하나는 y니까 
# 스플릿 함수로 나눠라



a = np.array(range(1, 101))

timesteps = 5   # x는 4개, y는 1개
# timesteps2 = 4 # x는4 y는 없어

def split_x(dataset, timesteps):
    aaa = []                                             # 출력할거 담을 빈 리스트
    for i in range(len(dataset) - timesteps + 1):  # 범위-타입스텝+1 만큼 반복    =>  (0, 1, 2) 가 i로 들어간다 = 0 
            subset = dataset[i : (i + timesteps)]
            aaa.append(subset)
    return np.array(aaa)
    
bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)    # (96, 5)




#슬라이싱
x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)
print(x.shape, y.shape)  # (96, 4) (96, )



x_predict = split_x(x_predict, timesteps)
print(x_predict)
print(x_predict.shape)    # (,)




 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234)
 
print(x_train.shape, y_train.shape)  # (,)
print(x_test.shape, y_test.shape)  # (,)

x_train = x.reshape(96,4,1)
x_test = x_test.reshape(24, 4, 1)
x_predict = x_predict.reshape(7,4,1)

print(x_train.shape, y_train.shape)  #
print(x_test.shape, y_test.shape)  #
print(x_predict.shape) #







#2. 모델구성
model = Sequential()   # (N, 5, 1) -> ([batch, timesteps, feature) = 타임스텝만큼 짜르고 피쳐만큼 일을 시킨다★
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
model.fit(x_train, y_train, epochs=1)


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

                                                    
result = model.predict(x_predict)


# [96,97,98,99]의 결과 :              을 만들자!







==============================================================================




import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from sklearn.model_selection import train_test_split
dataset = np.array(range(1,101))
x_predict = np.array(range(96,106))  # 예상 y = 100, 107

timesteps = 5     # x = 4개 y = 1개

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x(dataset, timesteps)


x = bbb[:, :-1]
y = bbb[:, -1]
x_predict = split_x(x_predict, 4)  # dataset , timestpes  
# x_predict = x_predict.reshape(7, 4, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      shuffle=True,
      train_size=0.75,
      random_state=123)

x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)
x_predict = x_predict.reshape(7,4,1)


# print (x,y)
# print (x.shape, y.shape) # (96,4)
# x = x.reshape(96,4,1)
# # x_predict = np.array(range(96,106))
# print(x_predict.shape)
# print(x.shape)

model = Sequential()
model.add(LSTM(64, input_shape=(4, 1), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# model.summary()

# 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=3)

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
result =model.predict(x_predict)
print('예측 결과 :', result)