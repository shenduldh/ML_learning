from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import SimpleRNN, Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.datasets import mnist
import numpy as np
np.random.seed(1337)  # for reproducibility


# 任务: 用 rnn 预测图片类型


# TIME_STEPS 指序列长度 (或时间点个数),
# 这里将图片的每一行像素点作为输入序列的元素,
# 图片高度为 28, 因此序列长度也为 28
TIME_STEPS = 28
# 这里表示每个时间点上输入数据的长度, 也就是图片的宽度
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train: shape(60000, 28, 28)
# y_train: shape(60000, )
# X_test: shape(10000, 28, 28)
# y_test: shape(10000, )

# preprocess data
X_train = X_train.reshape(-1, 28, 28) / 255.
X_test = X_test.reshape(-1, 28, 28) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # batch_input_shape=(batch_size, 序列长度, 输入数据的长度)
    # batch_size 在 tensorflow 中都用 None 表示
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,  # 输出数据的大小就是 rnn cell 的个数
    unroll=True,
    # 如果为 True，则网络将展开，否则将使用符号循环。
    # 展开可以加速 RNN，但它往往会占用更多的内存。
    # 展开只适用于短序列。
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# 编译模型
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# train
for step in range(4001):
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(
            X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost:', cost)
        print('test accuracy:', accuracy)
