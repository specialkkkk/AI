from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import sklearn as sk
from sklearn.datasets import load_boston



path = './_save/'
# path = '../_save/'
# path = 'c:/study/_save/'

# model.save(path + 'keras29_3_save_model.h5' )




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



print(x_train.shape, y_train.shape)  # (354, 13) (354,)
print(x_test.shape, y_test.shape)  # (152, 13) (152,)

x_train = x_train.reshape(354,13,1)
x_test = x_test.reshape(152, 13, 1)






#2. 모델구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(13, 1)))  
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))
model.summary()




# #2. 모델구성(함수형)
# input1 = Input(shape=(13,))
# dense1 = Dense(128, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(64, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(32, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(16, activation='relu')(drop3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()



 

 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = earlystopping = EarlyStopping(monitor='val_loss', patience=1, mode='min',
                              verbose=1, restore_best_weights=True)

import datetime
date = datetime.datetime.now()
print(date)    # 2023-01-12 14:58:02.348691
print(type(date))     # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   #0112_1457
print(date)    # 0112_1502
print(type(date))

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   #0037-0.0048.hdf5 




mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath= path  + 'MCP/keras30_ModelCheckPoint3.hdf5')        
                    filepath= filepath + 'k31_01_' + date + filename)




model.fit(x_train, y_train, epochs=1, batch_size=20,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)








#4. 평가,예측
print("==================== 1. 기본 출력 =================== ")
mse, mae = model.evaluate(x_test, y_test)
print('mse : ' , mse)


from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))



from sklearn.metrics import r2_score
print("RMSE : ",RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)



