import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Bidirectional
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
model.add(Bidirectional(LSTM(100), input_shape=(4,1)))
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