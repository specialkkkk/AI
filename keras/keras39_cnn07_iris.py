from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)       # 판다스 .describe()    /     .info()

'''
Y = 컬럼4개 input_dim=4
 
- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm
'''
print(datasets.feature_names)          # 판다스 .columns


x = datasets.data
y = datasets['target']
print(x)   
print(y)   
print(x.shape, y.shape)    # (150,4)  ,  (150,)
                           # x = 150짜리 1개 = 1
                           
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)



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
print(x_train.shape, x_test.shape) #(120, 4) (30, 4)




x_train = x_train.reshape(120, 4, 1, 1)
x_test = x_test.reshape(30, 4, 1, 1)
print(x_train.shape, x_test.shape) 





#2. 모델 구성
model = Sequential()
model.add(Conv2D(5, (2,1), 
                 input_shape=(4,1,1)))  # 내가 만든 인풋은 4,1이고 그것을 2,1(커널사이즈)로 쪼갠다
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(3, activation='softmax')) #아웃풋은 31번줄에 y값을 보고  

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2,
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

# accuracy :  0.93 Scaler

