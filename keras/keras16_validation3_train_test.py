import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array(range(1, 17))          # 1~ 16
y = np.array(range(1, 17))
# [실습] train_test_split만 사용해서 잘라라!!
# 10:3:3 으로 나눠라 !
# = [두번나눈다]  전체를 10:6으로 나누고 뒤에 3:3  6개를 반으로 나눔


print(x.shape)       #   16,

print(y.shape)       #    16,





x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.5,
    test_size=0.5,
    shuffle=False,
    random_state=123
)


print(x_train)
print(x_test)
print(y_train)
print(y_test)


x_validation = np.array([14,15,16])
y_validation = np.array([14,15,16])




#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation))



#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)


