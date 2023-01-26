import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
                                               #   . = 현재폴더(keras)   .. = 이전폴더
                                               #   path = '../_data/ddarung/'
                                               #   path = 'c:/study/_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

'''
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
'''

##### 결측치 처리 1. 제거 #####
# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# print(train_csv.shape)

x = train_csv.drop(['count'], axis=1)
# print(x)   # [1459 row x 9 columns]
y = train_csv['count']
# print(y)
# print(y.shape) # (1459,)



x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8, shuffle=True, random_state=123)
'''
print(x_train.shape, x_test.shape)   # (1021, 9)  (438, 9) 
print(y_train.shape, y_test.shape)   # (1021, )   (438,)
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape)   #(1062, 9) (266, 9)


x_train = x_train.reshape(1062, 9, 1, 1)
x_test = x_test.reshape(266, 9, 1, 1)
print(x_train.shape, x_test.shape)

#2. 모델구성
model = Sequential()
model.add(Conv2D(150, (2,1), input_shape=(9,1,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))



#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=10, restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=10,
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)




#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)




def RMSE(y_test, y_predict):                                    
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
print("RMSE:", RMSE(y_test, y_predict))
r2 = r2_score(y_test, y_predict)
print("R2:", r2)



#제출할 파일(submit)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)  # (715, 1)

# .to_csv()를 사용해서
# submission_0105.csv를 완성하시오! !


submission['count'] = y_submit
submission.to_csv(path + 'submission_0109 ddarung.csv')
print(submission)


print("RMSE : ", RMSE(y_test, y_predict))



print("=======================")
print(hist) #<keras.callbacks.History object at 0x0000024787770D90>
print("=======================")
print(hist.history)
print("=======================")
print(hist.history['loss'])
print("=======================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')

plt.title('ddarung loss')
plt.legend()

#plt.legend(loc='upper right')   = 오른쪽 위로 위치 지정

plt.show()


#RMSE : 51.763746   EarlyStopping


#RMSE : 43.045948   Scaler

