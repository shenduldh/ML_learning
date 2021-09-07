import numpy as np
import h5py


# 加载数据
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])

    train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))

    # 将样本数据进行扁平化和转置，以方便进行矩阵运算
    # 处理后的数组各维度的含义是（图片数据，样本数）
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # 对特征数据进行简单的标准化处理（除以255，使所有值都在[0，1]范围内）
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    return train_x, train_y, test_x, test_y


# 初始化参数
def initParams(layer_dims):
    """
    参数:
    layer_dims -- 数组，包含输入层特征数以及各层神经元数

    返回值:
    params -- 字典，存放初始化后的参数
    """
    layer_count = len(layer_dims)
    params = {}
    for i in range(1, layer_count):
        # 除以np.sqrt(layer_dims[i-1])，用于减小参数的数值
        params['W'+str(i)] = np.random.randn(layer_dims[i],
                                             layer_dims[i-1])/np.sqrt(layer_dims[i-1])
        params['b'+str(i)] = np.zeros((layer_dims[i], 1))

    return params


# 激活函数
def activate(Z, activation):

    if activation == 'relu':
        return np.maximum(0, Z)
    if activation == 'sigmoid':
        return 1/(1+np.exp(-Z))


# 前向传播：Z=WX+b, A=g(Z)
def propagateForward(X, params):

    layer_count = len(params)//2  # 整除以保证返回整数
    caches = []
    caches.append({
        'A': X
    })

    # 前layer_count-1层采用relu激活函数
    A = X
    for i in range(1, layer_count):
        A_prev = A
        Z = np.dot(params['W'+str(i)], A_prev)+params['b'+str(i)]
        A = activate(Z, activation='relu')
        caches.append({
            'Z': Z,
            'A': A
        })

    # 最后一层采用sigmoid激活函数
    Z_final = np.dot(params['W'+str(layer_count)], A) + \
        params['b'+str(layer_count)]
    A_final = activate(Z_final, activation='sigmoid')
    caches.append({
        'Z': Z_final,
        'A': A_final
    })

    return A_final, caches


# 计算成本
def compute_cost(A, Y):
    n = Y.shape[1]
    cost = (-1 / n) * np.sum(np.multiply(Y, np.log(A)) +
                             np.multiply(1 - Y, np.log(1 - A)))
    return cost


# 反向传播：从成本函数L开始
#   最后一层：dL/da_final, da_final/dz_final, dz_final/dw, dz/db
#   其他层：dz_prev/da, da/dz, dz/dw, dz/db
def propagateBackward(caches, Y,  params, learning_rate):

    layer_count = len(params)//2
    grads = {}

    # 求偏导数
    # 最后一层
    A_final = caches[-1]['A']
    n_final = A_final.shape[1]

    dA_final = np.divide(1-Y, 1-A_final)-np.divide(Y, A_final)
    dZ_final = dA_final * A_final*(1-A_final)
    dW_fianl = np.dot(dZ_final, caches[-2]['A'].T)/n_final
    db_final = np.sum(dZ_final, axis=1, keepdims=True)/n_final

    grads['dW'+str(layer_count)] = dW_fianl
    grads['db'+str(layer_count)] = db_final

    # 其他层
    dZ_back = dZ_final
    W_back = params['W'+str(layer_count)]
    for i in reversed(range(1, layer_count)):
        dA = np.dot(W_back.T, dZ_back)
        n = dA.shape[1]
        Z = caches[i]['Z']
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dW = np.dot(dZ, caches[i-1]['A'].T)/n
        db = np.sum(dZ, axis=1, keepdims=True)/n

        grads['dW'+str(i)] = dW
        grads['db'+str(i)] = db

        dZ_back = dZ
        W_back = params['W'+str(i)]

    # 更新参数
    for i in range(1, layer_count+1):
        params['W'+str(i)] -= learning_rate*grads['dW'+str(i)]
        params['b'+str(i)] -= learning_rate*grads['db'+str(i)]

    return params


def dnn_model(X, Y, layer_dims, learning_rate, times):

    params = initParams(layer_dims)

    for i in range(1, times+1):
        A_final, caches = propagateForward(X, params)
        params = propagateBackward(caches, Y, params, learning_rate)

        if(i % 100 == 0):
            cost = compute_cost(A_final, Y)
            print('第{i}次训练成本：{cost}'.format(i=i, cost=cost))

    return params


def predict(X, params):

    m = X.shape[1]
    p = np.zeros((1, m))

    A_final, caches = propagateForward(X, params)
    # 将预测结果转化成0和1的形式，即大于0.5的就是1，否则就是0
    for i in range(0, A_final.shape[1]):
        if A_final[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


train_x, train_y, test_x, test_y = load_data()
layer_dims = [12288, 20, 7, 5, 1]
params = dnn_model(train_x, train_y, layer_dims, 0.0075, 2000)

prediction = predict(test_x, params)
print("预测准确率是: " + str(np.sum((prediction == test_y) / test_x.shape[1])))
