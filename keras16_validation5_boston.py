#결과 RMSE :  4.009335467557085
#      R2  :  0.8233804556992295


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


import sklearn as sk
from sklearn.datasets import load_boston
print(sk.__version__)    # 1.1.3



#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)    # (506, 13)
print(y)
print(y.shape)    # (506,)

print(dataset.feature_names)
#'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT' 이런 컬럼=열=피처가 13개가 있다는 뜻.
print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True)

print(x_train.shape, x_test.shape) #(354,13) (152,13)
print(y_train.shape, y_test.shape) #(354,) (152,)



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
model.fit(x_train, y_train, epochs=200, batch_size=1,
          validation_split=0.25)


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

print("====================")
print(y_test)
print(y_predict)
print("=================")


def RMSE(y_test, y_predict):                                     
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
