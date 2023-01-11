from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


import sklearn as sk
from sklearn.datasets import load_boston



#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x)
scaler.transform(x)


print(x)
   




x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True)

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 



#2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=13))
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=20,
          validation_split=0.2,verbose=1)


#4. 평가,예측

mse, mae = model.evaluate(x_test, y_test)
print('mse : ' , mse)
print('mae : ' , mae)

from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))




from sklearn.metrics import r2_score



print("RMSE : ",RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
