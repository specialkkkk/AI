from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()             # 인풋은 (60000, 5, 5, 1) 행은 데이터의 개수 = 6만
model.add(Conv2D(filters=10, kernel_size=(2,2), # 필터= 처음 5x5에서 한번 돌릴때 4x4가 10개가 생긴다
                 input_shape=(5, 5, 1)))             # 아웃풋 = (N, 4, 4, 10)    행무시=None
model.add(Conv2D(5, kernel_size=(2,2)))        # 5 = 필터 생략한듯 (3, 3, 5)
model.add(Flatten())   # 펴주기   #(N, 45) 열=컬럼=특성=45개
model.add(Dense(10))  # (N, 10)
model.add(Dense(1))   # (N, 1)

model.summary()




