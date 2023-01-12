from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2),
                 input_shape=(5, 5, 1)))
model.add(Conv2D(5, kernel_size=(2,2)))
model.add(Flatten())   # 펴주기
model.add(Dense(10))
model.add(Dense(1))

model.summary()
