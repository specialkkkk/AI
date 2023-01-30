import numpy as np         
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)     # (100, 2)
                            
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).T
print(x2_datasets.shape)     # (100, 3)

x3_datasets = np.array([range(100,200), range(1301,1401)]).T
print(x3_datasets.shape)  # 

y1 = np.array(range(2001, 2101))  # (100,)  
y2 = np.array(range(201, 301))    # (100,) 





from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, \
        x3_train, x3_test, y1_train, y1_test, \
            y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.7, random_state=1234
)

print(x1_train.shape, x2_train.shape,x3_train.shape, y1_train, y2_train.shape) # (70, 2) (70, 3) (70,)
print(x1_test.shape, x2_test.shape,x3_test.shape, y1_test, y2_test.shape)  # (30, 3) (30, 3) (30,)

 
# 2. 모델1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

# 2. 모델2

input2 = Input(shape=(3,))
dense21 = Dense(21, activation='relu', name='ds21')(input2)
dense22 = Dense(22, activation='relu', name='ds22')(dense21)
output2 = Dense(23, activation='relu', name='ds23')(dense22)


# 3. 모델3

input3 = Input(shape=(2,))
dense31 = Dense(21, activation='relu', name='ds31')(input3)
dense32 = Dense(22, activation='relu', name='ds32')(dense31)
output3 = Dense(23, activation='relu', name='ds33')(dense32)



#2-3 모델병합
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2,output3], name='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)   
model = Model(inputs=[input1, input2, input3], outputs=last_output)



# 모델5 분기1
outputs=last_output
dense5 = Dense(22, activation='relu', name='ds51')(last_output)
dense5 = Dense(22, activation='relu', name='ds52')(dense5)
output5 =Dense(22, activation='relu', name='ds53')(dense5)

#모델6 분기2
outputs=last_output
dense6 = Dense(22, activation='relu', name='ds61')(last_output)
dense6 = Dense(22, activation='relu', name='ds62')(dense6)
output6 =Dense(22, activation='relu', name='ds63')(dense6)

model = Model(inputs=[input1, input2, input3], outputs=[output5,output6])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=10, batch_size=8)


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print('loss : ', loss)






