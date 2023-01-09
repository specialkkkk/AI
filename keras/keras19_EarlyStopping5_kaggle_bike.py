from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

#1. 데이터
path = './_data/bike/'
train_data = pd.read_csv(path + 'train.csv', index_col = 0)         # index_col = 0 → date_t 열 데이터로 취급 X
test_data = pd.read_csv(path + 'test.csv', index_col = 0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

# print(train_data.shape)          # (10886, 11) 
# print(test_data.shape)           # (6493, 8)
# print(train_data.columns)   
# print(train_data.info())         # Missing Attribute Values: 결측치 - 데이터에 값이 없는 것
# print(train_data.describe())   # 평균, 표준편차, 최대값 등

#  shape 맞추기 (열 제거) #
train_data = train_data.drop(['casual', 'registered'], axis = 1)

#  x,y 분리 #
x = train_data.drop(['count'], axis=1)                              # y 값(count 열) 분리, axis = 1 → 열에 대해 동작
y = train_data['count']                                             # y 값(count 열)만 추출

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=123
)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 8))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=5, restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)



#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) 
print('loss: ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE)

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)



# 제출
y_submit = model.predict(test_data)
submission['count'] = y_submit
submission.to_csv(path + 'samplesubmission_0109 bike.csv')

print("=======================")
print(hist) #<keras.callbacks.History object at 0x0000024787770D90>
print("=======================")
print(hist.history)
print("=======================")
print(hist.history['loss'])
print("=======================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('bike loss')
plt.legend()

#plt.legend(loc='upper right')   = 오른쪽 위로 위치 지정

plt.show()