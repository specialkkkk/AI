from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_breast_cancer()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)


x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) # (569, 30) (569,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2)



from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# #2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_shape=(30,)))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

#2. 모델구성(함수형)
input1 = Input(shape=(30,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()





#3.컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=20, restore_best_weights=True,
                              verbose=1)

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
                    filepath= filepath + 'k31_06_' + date + filename)
model.fit(x_train, y_train, epochs=100, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping,mcp],
          verbose=1)

#4. 평가 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', loss)

y_predict = model.predict(x_test)  #x_test를 넣었을때 y값을 예측할거다
y_predict = np.asarray(y_predict, dtype = int)
print(y_predict[:10])   #y 예측값 10개를 보자  = > 정수형으로 바꿔야 한다.
print(y_test[:10])      #y 원래값 10개를 보자





from sklearn.metrics import accuracy_score
# accuracy_score(y_test, y_predict)         # y테스트 값과  y예측값을 비교
# print("accuracy_score : ", acc)



                 



'''         
<정수로 바꾸기1>
y_predict = model.predict(x_test)
y_predict = np.asarray(y_predict, dtype = int)                 

# np.asarray: 입력된 데이터를 np.array 형식으로 만듬.      
                                          
# dtype 속성: 데이터 형식 변경 
(int: 정수형 / float: 실수형 / complex: 복소수형 / str: 문자형)
'''   


'''
<정수로 바꾸기2>
y_predict = y_predict.flatten()
y_predict = np.where(y_predict > 0.5, 1 , 0)

# preds_1d = y_predict.flatten() # 차원 펴주기
# pred_class = np.where(preds_1d > 0.5, 2 , 1) #0.5보다크면 2, 작으면 1


<정수로 바꾸기3>
round
 '''
 
 
