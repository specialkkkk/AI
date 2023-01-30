import numpy as np         
x_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x_datasets.shape)     # (100, 2)
                            


y1 = np.array(range(2001, 2101))  # (100,)  
y2 = np.array(range(201, 301))    # (100,) 





from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test,  y2_train, y2_test = train_test_split(
    x_datasets, y1, y2, train_size=0.7, random_state=1234
)

print(x_train.shape, y1_train, y2_train.shape) # (70, 2) (70, 3) (70,)
print(x_test.shape, y1_test, y2_test.shape)  # (30, 3) (30, 3) (30,)

 
# 2. 모델1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)





# 모델5 분기1
dense5 = Dense(22, activation='relu', name='ds51')(output1)
dense5 = Dense(22, activation='relu', name='ds52')(dense5)
output5 =Dense(22, activation='relu', name='ds53')(dense5)

#모델6 분기2
dense6 = Dense(22, activation='relu', name='ds61')(output1)
dense6 = Dense(22, activation='relu', name='ds62')(dense6)
output6 =Dense(22, activation='relu', name='ds63')(dense6)

model = Model(inputs=[input1], outputs=[output5,output6])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, [y1_train, y2_train], epochs=1, batch_size=8)


#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])
print('loss : ', loss)






