import matplotlib.pyplot as plt
# Dense: 用于构建全连接层
from tensorflow.keras.layers import Dense
# Sequential: 用于创建连续型 model, 即层与层之间按顺序构建的 model
from tensorflow.keras.models import Sequential
import numpy as np
np.random.seed(1337)  # for reproducibility


##################
# create some data
##################

X = np.linspace(-1, 1, 200)
# randomize the data
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]


##################
# build a neural network
##################

# 创建连续型 model
model = Sequential()
# 添加全连接层: 神经元个数为 1, 输入数据长度为 1
model.add(Dense(units=1, input_dim=1))
# 编译模型: 选择均方误差作为损失函数, 优化器为 SGD
model.compile(loss='mse', optimizer='sgd')


##################
# train, test, predict
##################

# training
print('****** Training *******')
for step in range(301):
    # 在 batch(X_train, Y_train) 上进行训练, 返回 cost
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost:', step, cost)

# test
print('****** Testing *******')
# 在测试集上进行评估, 返回 cost
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
# 获取 model 中第一层的参数
W, b = model.layers[0].get_weights()
print('Weights:', W)
print('biases:', b)

# plot the prediction
# 对某些东西进行预测, 返回预测值
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
