import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(y)
print(np.unique(y)) # [0 1 2] = y = 3
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

print(datasets.DESCR)      
print(datasets.feature_names)          # ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.9,
                                                    stratify=y)
print(y_train)
print(y_test)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(50, activation='relu', input_shape=(13,)))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='softmax'))             # 3 = y의 클래스의 개수

#2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
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
                    filepath= filepath + 'k31_08_' + date + filename)
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2,callbacks=[es, mcp],
          verbose=1)

#4. 평가,예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)   
print(acc)

# accuracy : 0.67 softmax

# accuracy : 0.86 Scaler