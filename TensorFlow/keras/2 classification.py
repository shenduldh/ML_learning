from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
np.random.seed(1337)  # for reproducibility


# download the mnist to the path '~/.keras/datasets/'
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train: shape(60000, 28, 28)
# y_train: shape(60000, )
# X_test: shape(10000, 28, 28)
# y_test: shape(10000, )

# preprocess data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
y_train = to_categorical(y_train, num_classes=10)  # to one_hot encoding
y_test = to_categorical(y_test, num_classes=10)  # to one_hot encoding

# build the neural network
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),  # 激活函数也当作一层
    Dense(10),  # 不需要输入数据的长度, 会自动从前一层中获取
    Activation('softmax')
])

# define the optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# 编译模型: 优化器采用自定义的 rmsprop, 损失函数采用交叉熵
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy']  # 表示同时输出预测的精确度 (accuracy)
              )

print('************ Training ************')
# 通过 fit 进行训练, epochs 表示训练次数
model.fit(X_train, y_train, epochs=2, batch_size=32)

print('************ Testing ************')
# 在测试集中对模型进行评估, 返回 cost 和 accuracy (由上面指定的需要额外输出的指标)
cost, accuracy = model.evaluate(X_test, y_test)
print('test loss:', cost)
print('test accuracy:', accuracy)
