from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

import sklearn as sk
from sklearn.datasets import load_boston



#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target


   

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True)


scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x)



#2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()



path = './_save/'
# path = '../_save/'
# path = 'c:/study/_save/'

 
 
 
 
 
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=20,
          validation_split=0.2,verbose=1)


model.save(path + 'keras29_3_save_model.h5' )
#R2 :  0.8781710546198362









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


# # mse :  16.1931095123291
# mae :  2.6015524864196777
# RMSE :  4.024066301770375
# R2 :  0.8120395136427374
