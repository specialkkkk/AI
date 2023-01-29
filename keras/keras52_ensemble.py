import numpy as np          # x1 삼성전자 시가  ,   x2 삼성전자 고가
import pandas as pd

PATH = 'C:/Users/SY/Desktop/'


x1_datasets = pd.read_csv(PATH + '삼성전자 주가.csv', header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(samsung)
# print(samsung.shape) #(1980, 17)

x2_datasets = pd.read_csv(PATH + '아모레퍼시픽 주가.csv', header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(amore)
# print(amore.shape)   #(2220, 17)








y = np.array(range(2001, 2101))  # (100,)   # 삼성전자의 하루뒤 종가

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=1234
)

print(x1_train.shape, x2_train.shape, y_train.shape) # (70, 2) (70, 3) (70,)
print(x2_test.shape, x2_test.shape, y_test.shape)  # (30, 3) (30, 3) (30,)
 
 
# 2. 모델1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

# 2. 모델2

input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)    # 여기서 1은 y를 말한다     y = np.array(range(2001, 2101))  # (100,)   y값이 1이었다.

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=8)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)





