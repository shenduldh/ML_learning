import numpy as np
import matplotlib.pyplot as plt


# x_data: shape(sample_count,1)
# y_data: shape(sample_count,1)
def load_data(sample_count):

    x_data = np.linspace(-1, 1, sample_count)[:, np.newaxis]
    bias = 0.5
    noise = np.random.normal(0, 0.1, x_data.shape)
    y_data = np.square(x_data)-bias + noise

    return x_data, y_data


# W: shape(in_size, out_size)
# b: shape(1, out_size)
def init_params(x_dim, layer_dims):

    layer_count = len(layer_dims)

    params = []

    for i in range(layer_count):
        if(i == 0):
            in_size = x_dim
        else:
            in_size = layer_dims[i-1]

        out_size = layer_dims[i]

        params.append({
            'W': np.random.randn(in_size, out_size),
            'b': np.zeros(shape=(1, out_size))
        })

    return params


# 激活函数
def relu(z):
    return np.maximum(0, z)


# 前向传播
def propagate_forward(x_data, params):

    layer_count = len(params)
    caches = [{'outputs': x_data}]  # 输入层

    outputs = x_data
    for i in range(layer_count):
        inputs = outputs
        WX_plus_b = np.dot(inputs, params[i]['W'])+params[i]['b']
        if(i == layer_count-1):  # 最后一层不使用激活函数
            outputs = WX_plus_b
        else:  # 其他层使用relu激活函数
            outputs = relu(WX_plus_b)
        caches.append({
            'WX_plus_b': WX_plus_b,
            'outputs': outputs
        })

    return outputs, caches


# 计算成本
def compute_cost(prediction, Y):
    cost = np.mean(np.square(prediction-Y))
    return cost


# 反向传播
def propagate_backward(caches, y_data,  params, learning_rate):

    layer_count = len(params)
    grads = []

    # 最后一层
    A_final = caches[-1]['outputs']
    n_final = A_final.shape[0]

    dA_final = A_final-y_data
    dZ_final = dA_final
    dW_fianl = np.dot(caches[-2]['outputs'].T, dZ_final)/n_final
    db_final = np.sum(dZ_final, axis=0, keepdims=True)/n_final

    grads.insert(0, {
        'dW': dW_fianl,
        'db': db_final
    })

    # 其他层
    dZ_back = dZ_final
    W_back = params[-1]['W']
    for i in reversed(range(layer_count-1)):
        dA = np.dot(W_back, dZ_back.T).T
        n = dA.shape[0]
        Z = caches[i+1]['WX_plus_b']
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dW = np.dot(caches[i]['outputs'].T, dZ)/n
        db = np.sum(dZ, axis=0, keepdims=True)/n
        grads.insert(0, {
            'dW': dW,
            'db': db
        })

        dZ_back = dZ
        W_back = params[i]['W']

    # 更新参数
    for i in range(layer_count):
        params[i]['W'] -= learning_rate*grads[i]['dW']
        params[i]['b'] -= learning_rate*grads[i]['db']

    return params


def DNN(x_date, y_data, layer_dims, learning_rate, iteration_times):

    params = init_params(x_date.shape[1], layer_dims)

    for i in range(iteration_times):
        prediction, caches = propagate_forward(x_date, params)
        params = propagate_backward(caches, y_data, params, learning_rate)

        if(i % 100 == 0):
            cost = compute_cost(prediction, y_data)
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            lines = ax.plot(x_data, prediction, 'r-', lw=3)
            plt.pause(0.5)
            print('第{i}次训练成本：{cost}'.format(i=i, cost=cost))

    return params


x_data, y_data = load_data(200)

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# 超参数
layer_dims = [3, 5, 1]
learning_rate = 0.1
iteration_times = 1000

DNN(x_data, y_data, layer_dims, learning_rate, iteration_times)

plt.pause(0)
