from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibility


# 任务: 翻译序列


BATCH_START = 0
TIME_STEPS = 20  # 序列长度
BATCH_SIZE = 50
INPUT_SIZE = 1  # 每个时间点输入每个 cell 的数据长度
OUTPUT_SIZE = 1  # 对每个时间点的输出降维为 1, 作为最终输出 (一共有 20 个时间点, 因此最终输出的长度为 20)
CELL_SIZE = 20  # LSTM cell 的个数为 20, 即每个时间点输出数据的长度为 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS *
                   BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


model = Sequential()

# build a LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
    units=CELL_SIZE,  # LSTM cell 的个数, 也是每个时间步的输出长度
    # return_sequences 用于指定输出结果的形式
    # 为 True: 表示输出所有时间步的输出
    # 为 False: 表示仅输出最后一个时间步的输出 (默认)
    return_sequences=True,
    # stateful 表示 batch 之间的 state 是否具有关联
    # 为 True: 将前一个 batch 的最终 state 作为下一个 batch 的初始 state
    # 默认为 False, 即 batch 之间没有状态的联系
    stateful=True
))

# add output layer
# TimeDistributed 表示对 rnn 中每个时间步的输出都用一个 Dense (全连接层) 进行操作
# 这些 Dense 的操作就是将每个时间步输出的数据的长度降为 1 (即 OUTPUT_SIZE),
# 然后将每个时间步降维后的数据合并为最终的输出,
# 一共有 20 个时间点, 因此最终输出的长度为 20
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

model.compile(optimizer=Adam(LR), loss='mse')


print('************ Training ************')
for step in range(501):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r',
             xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost:', cost)
