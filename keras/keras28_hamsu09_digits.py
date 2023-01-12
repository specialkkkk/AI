import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)             = 1797행과  /  64개의 컬럼 (열)  = input
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# output =10 (0 1 2 3 4 5 6 7 8 9)

'''
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[10])
plt.show()
'''

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.8,
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
# model.add(Dense(110, activation='relu', input_shape=(64,)))
# model.add(Dense(90, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='softmax'))            

#2. 모델구성(함수형)
input1 = Input(shape=(64,))
dense1 = Dense(110, activation='relu')(input1)
dense2 = Dense(90, activation='relu')(dense1)
dense3 = Dense(60, activation='relu')(dense2)
dense4 = Dense(50, activation='relu')(dense3)
dense5 = Dense(30, activation='relu')(dense4)
output1 = Dense(10, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=20, restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4. 평가,예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)


from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)

acc = accuracy_score(y_test, y_predict)   
print(acc)

# accuracy : 0.92 Scaler