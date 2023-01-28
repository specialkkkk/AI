import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, LSTM
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
print(y)                         #Y =10
print(y.shape)




x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2,
                                                    stratify=y)
print(y_train)
print(y_test)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(1437, 64) (360, 64)



x_train = x_train.reshape(1437, 8, 8)
x_test = x_test.reshape(360, 8, 8)
print(x_train.shape, x_test.shape) 







#2. 모델 구성
model = Sequential()
model.add(LSTM(units=5, input_shape=(8,8)))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(10, activation='softmax'))  #아웃풋= Y값

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


