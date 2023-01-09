from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss' , 
                              mode='min', 
                              patience=20, restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4. 평가 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', loss)

y_predict = model.predict(x_test)  #x_test를 넣었을때 y값을 예측할거다
print(y_predict[:10])   #y 예측값 10개를 보자  = > 정수형으로 바꿔야 한다.
print(y_test[:10])      #y 원래값 10개를 보자
'''
[9.9832189e-01]
[1.5013814e-01]
[9.9996424e-01]
[9.6233487e-01]
[2.4487136e-08]
[9.9998665e-01]
[8.5847747e-01]
[9.9963391e-01]
[3.1936169e-04]
[9.9168044e-01]
[1 0 1 1 0 1 1 1 0 1]
print 결과 마이너스(실수)가 나왔다 정수로 바꿔야함.
'''
y_predict = y_predict.flatten()
y_predict = np.where(y_predict > 0.5, 1 , 0)
 
from sklearn.metrics import accuracy_score
# accuracy_score(y_test, y_predict)         # y테스트 값과  y예측값을 비교
# print("accuracy_score : ", acc)


