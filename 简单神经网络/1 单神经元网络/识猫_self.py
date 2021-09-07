import numpy as np  # 用于数据处理和运算
import h5py  # 用于读取格式为h5的数据集

import matplotlib.pyplot as plt  # 显示图片和曲线
import skimage.transform as tf  # 转换图片


# 加载和处理数据
def loadData():
    # 读取类型为h5的训练数据和测试数据
    train_dataset = h5py.File("./datasets/train_catvnoncat.h5", "r")
    test_dataset = h5py.File("./datasets/test_catvnoncat.h5", "r")

    # 读取标签类型，0表示'是猫'，1表示'不是猫'
    # NOTE a[:]等同于a[0:]
    classes = np.array(train_dataset['list_classes'][:])
    print(classes[0], classes[1])

    # 分别读取输入数据和标签数据，并转化为矩阵
    train_set_x = np.array(train_dataset['train_set_x'][:])
    test_set_x = np.array(test_dataset['test_set_x'][:])
    train_set_y = np.array(train_dataset['train_set_y'][:])
    test_set_y = np.array(test_dataset['test_set_y'][:])
    # 转换成行向量以方便后续计算
    train_set_y = train_set_y.reshape(1, train_set_y.shape[0])
    test_set_y = test_set_y.reshape(1, test_set_y.shape[0])

    # 训练输入图片样本包含四个维度：(209, 64, 64, 3)
    # 分别表示：209个样本、宽度和高度均为64、每个像素包含3个RGB通道
    # 理解：train_set_x表示一个包含209个元素（即样本）的数组，每个元素的值是一个平面集（二维数组），
    #       平面集上每个点是一个包含3个元素的数组（即包含3个RGB通道的像素）
    print("train_set_x shape：", train_set_x.shape)
    # train_set_y为包含209个元素（样本标签）的一维数组
    print("train_set_y shape：", train_set_y.shape)

    # NOTE reshape中的参数-1被理解为unspecified value。如果我只需要特定的行数，列数多少无所谓，
    # 则只需要指定行数，而列数用-1表示即可，由计算机帮我们计算列数，反之亦然。
    # NOTE numpy.ndarray.T：转置矩阵。
    # 此处将4维样本数据转化为2维数组（209×12288矩阵），即包含209个元素的数组，
    # 其元素为包含某个样本图片的所有像素3个RGB通道的一维数组（有64*64*3=12288个元素）。
    # 最后进行转置T，变成12288×209矩阵，每一个列向量是一个样本。
    train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T
    test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T
    # 将所有像素的3个RGB通道值除以255，以减小输入数据的大小。
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    return train_set_x, train_set_y, test_set_x, test_set_y


# 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 初始化权重和阈值
def initParams(dim):
    # 此处dim=12288，即一个RGB通道对应一个权重
    w = np.zeros((dim, 1))
    b = 0
    return w, b


# 一次传播包括前向传播和反向传播。
# 并非是一个样本传播一次，然后通过大量样本来优化参数。
# ⭐而是一次传播就将所有样本数据一起计算，通过不断重复这一过程来达到优化参数的目的。
def propagate(w, b, X, Y):
    """
    w：(12288, 1)列向量，一个元素代表一个RGB通道的权重
    b：一个数值，阈值
    X：(12288, x)12288维列向量组成的矩阵，每个列向量代表一个样本（包含12288个RGB通道）
    Y：(1, y)行向量，每个元素代表一个标签
    """
    m = X.shape[1]  # 样本数
    # 前向传播
    A = sigmoid(np.dot(w.T, X) + b)  # 209维行向量：每个元素代表1个样本的预测值
    cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / m  # 计算所有样本的损失值，然后求和算平均值
    # 反向传播
    dZ = A - Y  # 两个行向量相减
    dw = np.dot(X, dZ.T) / m  # 求w的平均梯度
    db = np.sum(dZ) / m  # 求b的平均梯度
    # 将dw和db保存到字典里面
    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost


# 参数优化
def optimize(w, b, X, Y, times, learning_rate):
    costs = []
    # 重复优化times次
    for i in range(times):
        grads, cost = propagate(w, b, X, Y)
        # 梯度下降
        w -= learning_rate*grads["dw"]
        b -= learning_rate*grads["db"]
        # 记录每100次优化的损失值
        if i % 100 == 0:
            costs.append(cost)
            print("第%i次优化的成本是：%f" % (i, cost))
    params = {
        "w": w,
        "b": b
    }
    return params, costs


# 预测
def predict(w, b, X):
    m = X.shape[1]
    prediction = np.zeros((1, m))  # 将所有预测值初始化为0
    A = sigmoid(np.dot(w.T, X) + b)  # 行向量：每个元素代表1个样本的预测值
    # 如果预测值大于0.5，则表示'有猫'并将预测结果设为1
    for i in range(m):
        # NOTE A[0,i]表示第0行第i列的元素
        if A[0, i] > 0.5:
            prediction[0, i] = 1
    return prediction


# 整合学习过程
def model(x_train, y_train, x_test, y_test, times=2000, learning_rate=0.005):
    # 初始化参数
    w, b = initParams(x_train.shape[0])
    # 优化参数
    params, costs = optimize(w, b, x_train, y_train, times, learning_rate)
    # 将参数设置为优化后的值
    w = params["w"]
    b = params["b"]

    # 对测试集进行预测
    test_prediction = predict(w, b, x_test)
    # 打印结果
    print("对测试集的预测准确率为: {}%".format(
        100 - np.mean(np.abs(test_prediction - y_test)) * 100))

    # 返回优化产生的模型
    return {
        "costs": costs,
        "params": params,
        "test_prediction": test_prediction,
        "learning_rate": learning_rate,
        "times": times
    }


train_set_x, train_set_y, test_set_x, test_set_y = loadData()
model = model(train_set_x, train_set_y, test_set_x, test_set_y)


# 显示某一张测试图片及其预测结果
index = 5
# 取出第index列的列向量，即第index个样本的所有像素数据
# 然后将其重塑为3维数组：64×64的平面集，每个点包含3个RGB通道值
plt.imshow(test_set_x[:, index].reshape((64, 64, 3)))
print("测试样本%i的预测结果是：%i，标签是%f" %
      (index, model["test_prediction"][0, index], test_set_y[0, index]))
# np.squeeze用于去除shape中为1的维度
costs = np.squeeze(model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(model["learning_rate"]))
plt.show()


# 使用model预测实际图片
myImage_url = 'images/test1.jpg'
myImage_data = np.array(plt.imread(myImage_url))  # 3维数组(高度, 宽度, 3)
# 用tf将图片重塑为(64, 64, 3)的3维数组，进而转化为(12288,1)列向量
myImage = tf.resize(myImage_data, (64, 64),
                    mode='reflect').reshape((1, 64*64*3)).T
myImage_prediction = predict(
    model["params"]["w"], model["params"]["b"], myImage)
print("对%s图片的预测结果为：%i" % (myImage_url, myImage_prediction))
plt.imshow(myImage_data)
plt.show()
