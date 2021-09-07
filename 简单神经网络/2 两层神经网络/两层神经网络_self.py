# 目标模型：预测直角坐标轴上某点的颜色
import numpy as np


# 加载训练集（模拟数据）
def load_dataset():
    np.random.seed(1)
    m = 400  # 样本数
    N = int(m/2)  # number of points per class
    D = 2  # 横、纵坐标
    X = np.zeros((m, D))  # data matrix where each row is a single example
    # labels vector (0 for red, 1 for blue)
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + \
            np.random.randn(N)*0.2  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2  # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T  # 每列向量->一个样本
    Y = Y.T  # 行向量，每个元素->一个样本预测值

    return X, Y


# 初始化参数
def init_params(one_x, one_y, two_y):
    # W1：行向量矩阵，每个行向量->一个神经元的权重
    W1 = np.random.randn(one_y, one_x)*0.01
    # B1：列向量，每个元素->一个神经元的阈值
    B1 = np.zeros(shape=(one_y, 1))
    # 同上
    W2 = np.random.randn(two_y, one_y)*0.01
    # 同上
    b2 = np.zeros(shape=(two_y, 1))

    return {"W1": W1,
            "B1": B1,
            "W2": W2,
            "b2": b2}


# sigmoid激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))


# 前向传播
def forward_propagate(X, Y, params):
    W1 = params["W1"]
    B1 = params["B1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(W1, X)+B1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    # np.multipy：对应元素相乘
    n = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(A2))+np.multiply(1-Y, np.log(1-A2)))/n

    return cost, {"Z1": Z1,
                  "A1": A1,
                  "Z2": Z2,
                  "A2": A2}


# 反向传播
def backward_propagate(params, result, X, Y):
    n = X.shape[1]  # 获取样本数

    # W1 = params['W1']
    W2 = params['W2']

    A1 = result['A1']
    A2 = result['A2']

    # 计算偏导数
    dZ2 = A2 - Y
    dW2 = (1 / n) * np.dot(dZ2, A1.T)
    db2 = (1 / n) * np.sum(dZ2)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / n) * np.dot(dZ1, X.T)
    dB1 = (1 / n) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1,
            "dB1": dB1,
            "dW2": dW2,
            "db2": db2}


# 更新参数
def update_params(params, grads, learning_rate=0.5):
    W1 = params["W1"]
    B1 = params["B1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = grads['dW1']
    dB1 = grads['dB1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return {"W1": W1,
            "B1": B1,
            "W2": W2,
            "b2": b2}


# 预测
def predict(Y, params):
    W1 = params["W1"]
    B1 = params["B1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(W1, Y)+B1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    return np.round(A2)


# 模型
def model(X, Y, neurons_num, times, learning_rate=0.5):
    one_x = X.shape[0]
    two_y = Y.shape[0]
    params = init_params(one_x, neurons_num, two_y)

    for i in range(times):
        cost, result = forward_propagate(X, Y, params)
        grads = backward_propagate(params, result, X, Y)
        params = update_params(params, grads)
        if((i+1) % 100 == 0):
            print("第{}次优化cost：{}".format(i+1, cost))

    return params


X, Y = load_dataset()
params = model(X, Y, 4, 2000)
predictions = predict(X, params)
print('预测准确率是: %d' % float((np.dot(Y, predictions.T) +
                            np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
