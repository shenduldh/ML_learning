import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def add_layer(inputs, in_size, out_size, activator=None):
    """
        inputs: [1, in_size]
        in_size: 每个神经元的权重个数
        out_size: 神经元数目, 有多少个神经元就有多少个输出
        activator: 激励函数
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    if activator is None:
        outputs = Wx_plus_b
    else:
        outputs = activator(Wx_plus_b)
    return outputs


###############
## 创建数据集 ##
##############
# np.linspace: 创建从-1开始到1结束的包含300个项的等差数列
# 指定类型为float32原因:
#       tf默认创建的变量类型为float32, 而且tf的运算要求参数类型一致,
#       因此先将数据类型转化为float32, 以避免后续类型不一致的问题。
#       此外, 方法tf.cast()可以用于数据类型的转化
# [:,np.newaxis]: 在原来数据的基础上增加一个维度
# x_data.shape == [300, 1]
x_data = np.linspace(-1, 1, num=300, dtype=np.float32)[:, np.newaxis]  # 特征
# 给数据增加噪点
# np.random.normal: 以正态分布 (均值为0, 标准差为0.05) 随机初始值, 形状为x_data.shape
# .astype(np.float32): 转换数据类型为float32, 因为np默认创建的数据类型为float64
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise  # 标签


#############
## 构建模型 ##
############
# 创建占位符, [None, 1]表示行数不确定, 由具体输入值决定
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

layer_one = add_layer(xs, 1, 10, activator=tf.nn.relu)

prediction = add_layer(layer_one, 10, 1, activator=None)

loss = tf.reduce_mean(tf.reduce_sum(
    tf.square(prediction-ys), axis=1))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


#############
## 训练模型 ##
############
sess = tf.Session()
sess.run(init)

figure = plt.figure()  # 创建一个画板
# 在指定位置生成一个数轴，返回该数轴对象axes
# 前两个参数用于对figure进行网格划分，这里将其划分为1行1列的网格
# 最后一个参数指将这个数轴放在第一个网格中
ax = figure.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)  # 创建散点图
plt.ion()  # 表示画图时不暂停程序
plt.show()  # 将figure展现出来

for step in range(1001):
    sess.run(train, feed_dict={
        xs: x_data,
        ys: y_data
    })
    if step % 50 == 0:
        # print(step, sess.run(loss, feed_dict={
        #     xs: x_data,
        #     ys: y_data
        # }))
        try:
            ax.lines.remove(lines[0])  # 抹除lines中的第一个线段
        except:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=3)  # 画曲线
        plt.pause(0.5)  # 暂停0.5s

plt.pause(0)  # 一直暂停，直到点叉
