import numpy as np  # 用于数据处理和运算
import h5py  # 用于读取格式为h5的数据集

import matplotlib.pyplot as plt  # 显示图片和曲线
import skimage.transform as tf  # 转换图片


# 加载数据
# X: shape(209, 12288)
# Y: shape(209, 1)
def loadData():
    train_dataset = h5py.File("./datasets/train_catvnoncat.h5", "r")
    test_dataset = h5py.File("./datasets/test_catvnoncat.h5", "r")

    train_set_x = np.array(train_dataset['train_set_x'][:])
    test_set_x = np.array(test_dataset['test_set_x'][:])
    train_set_y = np.array(train_dataset['train_set_y'][:])
    test_set_y = np.array(test_dataset['test_set_y'][:])

    train_set_y = train_set_y.reshape(train_set_y.shape[0], 1)
    test_set_y = test_set_y.reshape(test_set_y.shape[0], 1)

    train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1)
    test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1)

    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    return train_set_x, train_set_y, test_set_x, test_set_y


# 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 初始化权重和阈值
def initParams(dim):
    W = np.zeros((dim, 1))
    b = 0
    return W, b


# 前向传播和反向传播
def propagate(W, b, X, Y):
    """
    W: shape(12288, 1)
    b: shape(1, )
    X: shape(209, 12288)
    Y: shape(209, 1)
    """

    m = X.shape[0]  # 样本数

    # 前向传播
    Z = np.dot(X, W) + b  # shape(209, 1)
    A = sigmoid(Z)  # shape(209, 1)
    cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / m  # shape(1, )

    # 反向传播
    dZ = A - Y
    dW = np.dot(X.T, dZ) / m
    db = np.sum(dZ) / m

    grads = {
        "dW": dW,
        "db": db
    }

    return grads, cost


# 梯度下降
def optimize(W, b, X, Y, times, learning_rate):
    costs = []

    for i in range(times):
        grads, cost = propagate(W, b, X, Y)

        W -= learning_rate*grads["dW"]
        b -= learning_rate*grads["db"]

        if i % 100 == 0:
            costs.append(cost)
            print("第%i次优化的成本是：%f" % (i, cost))

    params = {
        "W": W,
        "b": b
    }

    return params, costs


# 预测
def predict(W, b, X):
    m = X.shape[0]

    prediction = np.zeros((m, 1))
    A = sigmoid(np.dot(X, W) + b)

    for i in range(m):
        if A[i, 0] > 0.5:
            prediction[i, 0] = 1

    return prediction


# 整合
def model(x_train, y_train, x_test, y_test, times=2000, learning_rate=0.005):

    W, b = initParams(x_train.shape[1])

    params, costs = optimize(W, b, x_train, y_train, times, learning_rate)

    W = params["W"]
    b = params["b"]

    test_prediction = predict(W, b, x_test)

    print("对测试集的预测准确率为: {}%".format(
        100 - np.mean(np.abs(test_prediction - y_test)) * 100))

    return {
        "costs": costs,
        "params": params,
        "test_prediction": test_prediction,
        "learning_rate": learning_rate,
        "times": times
    }


train_set_x, train_set_y, test_set_x, test_set_y = loadData()
model = model(train_set_x, train_set_y, test_set_x, test_set_y)


# 使用model预测实际图片
my_image_url = 'images/test.jpg'
my_image_data = np.array(plt.imread(my_image_url))

my_image = tf.resize(my_image_data, (64, 64),
                     mode='reflect').reshape((1, 64*64*3))
my_image_prediction = predict(
    model["params"]["W"], model["params"]["b"], my_image)
print("对%s图片的预测结果为：%i" % (my_image_url, my_image_prediction))

plt.imshow(my_image_data)
plt.show()
