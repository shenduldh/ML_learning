import numpy as np
import matplotlib.pyplot as plt
import h5py
import skimage.transform as tf


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  # 加载训练数据
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:])  # 从训练数据中提取出图片的特征数据
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # 从训练数据中提取出图片的标签数据

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")  # 加载测试数据
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # 加载标签类别数据，这里的类别只有两种，1代表有猫，0代表无猫
    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(
        (1, train_set_y_orig.shape[0]))  # 把数组的维度从(209,1)变成(1, 209)，这样好方便后面进行计算
    test_set_y_orig = test_set_y_orig.reshape(
        (1, test_set_y_orig.shape[0]))  # 从(50,1)变成(1, 50)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_orig shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# 上面train_set_x_orig的各维度的含义分别是(样本数，图片宽，图片长，3个RGB通道)
# 我们后面要用到样本数和长宽像素值，下面我分别把它们提取出来了。

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]  # 由于我们的图片是正方形的，所以长宽相等

print("训练样本数: m_train = " + str(m_train))
print("测试样本数: m_test = " + str(m_test))
print("每张图片的宽/高: num_px = " + str(num_px))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


def sigmoid(z):
    """
    参数:
    z -- 一个数值或者一个numpy数组.
    返回值:
    s -- 经过sigmoid算法计算后的值，在[0,1]范围内
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    这个函数用于初始化权重数组w和偏置/阈值b.

    参数:
    dim -- w的大小，看上面的神经网络模型图可知，dim在本例中是12288，因为一个特征输入对应一个权重。

    返回值:
    w -- 权重数组
    b -- 偏置bias
    """

    w = np.zeros((dim, 1))
    b = 0

    return w, b


def propagate(w, b, X, Y):
    """
    参数:
    w -- 权重数组，维度是(12288, 1)
    b -- 偏置bias
    X -- 图片的特征数据，维度是 (12288, 209)
    Y -- 图片对应的标签，0或1，0是无猫，1是有猫，维度是(1,209)

    返回值:
    cost -- 成本
    dw -- w的梯度
    db -- b的梯度
    """

    m = X.shape[1]

    # 前向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / m

    # 反向传播
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m

    # 将dw和db保存到字典里面
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """    
    参数:
    w -- 权重数组，维度是 (12288, 1)
    b -- 偏置bias
    X -- 图片的特征数据，维度是 (12288, 209)
    Y -- 图片对应的标签，0或1，0是无猫，1是有猫，维度是(1,209)
    num_iterations -- 指定要优化多少次
    learning_rate -- 学习步进，是我们用来控制优化步进的参数
    print_cost -- 为True时，每优化100次就把成本cost打印出来,以便我们观察成本的变化

    返回值:
    params -- 优化后的w和b
    costs -- 每优化100次，将成本记录下来，成本越小，表示参数越优化
    """

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)  # 计算得出梯度和成本

        # 从字典中取出梯度
        dw = grads["dw"]
        db = grads["db"]

        # 进行梯度下降，更新参数，使其越来越优化，使成本越来越小
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 将成本记录下来
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("优化%i次后成本是: %f" % (i, cost))

    params = {"w": w,
              "b": b}
    return params, costs


def predict(w, b, X):
    '''    
    参数:
    w -- 权重数组，维度是 (12288, 1)
    b -- 偏置bias
    X -- 图片的特征数据，维度是 (12288, 图片张数)

    返回值:
    Y_prediction -- 对每张图片的预测结果
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)  # 通过这行代码来对图片进行预测

    # 上面得出的预测结果是小数的形式，为了方便后面显示，我们将其转换成0和1的形式（大于等于0.5就是1/有猫，小于0.5就是0/无猫）
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """    
    参数:
    X_train -- 训练图片,维度是(12288, 209)
    Y_train -- 训练图片对应的标签,维度是 (1, 209)
    X_test -- 测试图片,维度是(12288, 50)
    Y_test -- 测试图片对应的标签,维度是 (1, 50)
    num_iterations -- 需要训练/优化多少次
    learning_rate -- 学习步进，是我们用来控制优化步进的参数
    print_cost -- 为True时，每优化100次就把成本cost打印出来,以便我们观察成本的变化

    返回值:
    d -- 返回一些信息
    """

    # 初始化待训练的参数
    w, b = initialize_with_zeros(X_train.shape[0])

    # 使用训练数据来训练/优化参数
    parameters, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从字典中分别取出训练好的w和b
    w = parameters["w"]
    b = parameters["b"]

    # 使用训练好的w和b来分别对训练图片和测试图片进行预测
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 打印出预测的准确率
    print("对训练图片的预测准确率为: {}%".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("对测试图片的预测准确率为: {}%".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)

index = 5
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print("这张图的标签是 " + str(test_set_y[0, index]) +
      ", 预测结果是 " + str(int(d["Y_prediction_test"][0, index])))

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')  # 成本
plt.xlabel('iterations (per hundreds)')  # 横坐标为训练次数，以100为单位
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("学习率为: " + str(i) + "时")
    models[str(i)] = model(train_set_x, train_set_y, test_set_x,
                           test_set_y, num_iterations=1500, learning_rate=i, print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),
             label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

my_image = "test.jpg"
fname = "images/" + my_image

image = np.array(plt.imread(fname))
my_image = tf.resize(image, (num_px, num_px),
                     mode='reflect').reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("预测结果为 " + str(int(np.squeeze(my_predicted_image))))
