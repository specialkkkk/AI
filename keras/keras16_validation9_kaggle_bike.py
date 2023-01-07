import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)  # (10886, 11)    #


print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())



##### 결측치 처리 1. 제거 #####
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)




x = train_csv.drop(['count','casual','registered'], axis=1)
print(x)   # [10886 rows x 10 columns]
y = train_csv['count']
print(y)
print(y.shape) # (10886,)




x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=209)

print(x_train.shape, x_test.shape)   # (7620, 10) (3266, 10)
print(y_train.shape, y_test.shape)   # (7620,) (3266,)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))




#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start= time.time()
model.fit(x_train, y_train, epochs=200, batch_size=10)
end = time.time()
#배치 32  149




#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)




def RMSE(y_test, y_predict):                                      # #predict : 예측값 y_test  y나눠진값.
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print("RMSE:", RMSE(y_test, y_predict))
r2 = r2_score(y_test, y_predict)
print("R2:", r2)



#제출할 파일(submit)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)  # (715, 1)





submission['count'] = y_submit
submission.to_csv(path + 'submission_20230106_1.csv')
print(submission)


print("RMSE : ", RMSE(y_test, y_predict))

print("걸린시간 : ", end - start)



# 150.
