# [실습]
# R2 0.55~0.6 이상 만들어라.

#0.63

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(x.shape) #(20640, 8)
print(y)
print(y.shape) #(20640,)

print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123
)

#2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))

#결과 R2 : 0.548836

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

print("====================")
print(y_test)
print(y_predict)
print("=================")


 
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
