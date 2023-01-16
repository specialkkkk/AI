import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() #이미 트레인,테스트 분리가 되어있다.

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28)  (10000,)

print(x_train[1000])
print(y_train[1000])  # 5

import matplotlib.pyplot as plt
plt.imshow(x_train[1000], 'gray')
plt.show()

