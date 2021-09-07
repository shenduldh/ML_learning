from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.datasets import mnist
import numpy as np
np.random.seed(1337)  # for reproducibility


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train: shape(60000, 28, 28)
# y_train: shape(60000, )
# X_test: shape(10000, 28, 28)
# y_test: shape(10000, )

# preprocess data
X_train = X_train.reshape(-1, 1, 28, 28)/255.
X_test = X_test.reshape(-1, 1, 28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 构建连续型 model
model = Sequential()

# Conv layer 1
# output shape (-1, 32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,  # 卷积核个数, 也是输出的 chanel 数
    kernel_size=5,  # 卷积核大小
    strides=1,  # 步长
    padding='same',  # 填充方法
    data_format='channels_first',  # 表示输入数据的 chanel 维度放在首位
))
model.add(Activation('relu'))  # 使用 relu 进行激活

# Pooling layer 1
# output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,  # 池化核大小
    strides=2,  # 步长为2, 表示长宽缩小一半
    padding='same',  # 填充方法
    data_format='channels_first',
))

# Conv layer 2
# output shape (64, 14, 14)
# batch_input_shape 由 Keras 自动根据上层决定
model.add(Convolution2D(
    64, 5,  # 分别指 64 个卷积核, 卷积核大小为 5
    strides=1,
    padding='same',
    data_format='channels_first'
))
model.add(Activation('relu'))

# Pooling layer 2
# output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1
# input shape (64 * 7 * 7, ) = (3136), output shape (1024, )
model.add(Flatten())  # 将三维的输入数据抹平成一维数据
model.add(Dense(1024))  # 1024 指神经元个数
model.add(Activation('relu'))

# Fully connected layer 2
# output shape (10, ) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# define the optimizer
adam = Adam(lr=1e-4)

# 编译模型
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print('************ Training ************')
model.fit(X_train, y_train, epochs=1, batch_size=64)

print('************ Testing ************')
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss:', loss)
print('test accuracy:', accuracy)
