from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data               # (506, 13)
y = datasets.target             # (506,)

print(x.shape, y.shape)         # (506, 13) (506,) 행무시 열(13)우선

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(8, input_shape=(13,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss' ,                 #earlyStopping 맨앞 소문자는 임의로 만든것
                              mode='min', 
                              patience=5, restore_best_weights=True,
                              verbose=1)              

hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1)


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)


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
plt.title('보스톤 손실량')
plt.legend()

#plt.legend(loc='upper right')   = 오른쪽 위로 위치 지정

plt.show()








