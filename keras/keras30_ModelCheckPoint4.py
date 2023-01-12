from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
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


print(x)



#2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(16, activation='relu')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()



 
 
 
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
                    filepath= filepath + 'k30_' + date + filename)




model.fit(x_train, y_train, epochs=1000, batch_size=20,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)


model.save(path + "keras30_ModelCheckPoint3_save_mode.h5")





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



