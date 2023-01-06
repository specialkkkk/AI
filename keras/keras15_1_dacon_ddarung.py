import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)  # (1459, 10)    #id빼고 9개

print(train_csv.columns)
#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#   dtype='object')
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

x = train_csv.drop(['count'], axis=1)
print(x)   # [1459 row x 9 columns]
y = train_csv['count']
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234)

print(x_train.shape, x_test.shape)   # (1021, 9)  (438, 9) 
print(y_train.shape, y_test.shape)   # (1021, )   (438,)


#2. 모델구성
model = Sequential()
model.add(Dense(10,input_dim=9))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 나쁨!!
# 결측치 때문에 To Be Countinue



def RMSE(y_test, y_predict):                                      # #predict : 예측값 y_test  y나눠진값.
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

#제출할 파일(submit)
y_submit = model.predict(test_csv)
