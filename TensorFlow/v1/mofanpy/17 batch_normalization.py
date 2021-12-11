import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()


ACTIVATOR = tf.nn.relu
LAYER_COUNT = 7
N_HIDDEN_UNITS = 30


# 使每次产生的随机数相同
def fix_seed(seed=1):
    np.random.seed(seed)
    tf.set_random_seed(seed)


# 显示输出分布
# 横轴表示输出值, 纵轴表示对应输出值的数量
def plot_inputs(inputs, inputs_norm):
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.1)


def build_net(x, y, norm):
    def add_layer(inputs, in_size, out_size, norm, activator=None):
        Weights = tf.Variable(tf.random_normal(
            [in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # 对每层激活前的输出进行批标准化
        if norm:
            # 求出均值和方差, 用于后续的标准化
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                # 指定normalize的维度, [0]代表batch维度
                # 如果是图像数据, 可以传入[0, 1, 2], 相当于求[batch, height, width]的均值和方差,
                # 注意不要加入channel维度, 即不要加入实际被运算的维度
                axes=[0]
            )
            # 用于反标准化的模型参数
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001  # 防止除以零
            Wx_plus_b = tf.nn.batch_normalization(
                Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
            # 上面最后一步相当于下面两句代码:
            # Wx_plus_b = (Wx_plus_b - mean) / tf.sqrt(var + epsilon)
            # Wx_plus_b = Wx_plus_b * scale + shift

            # 如果是使用MiniBatch进行训练, 则每个batch的mean和var都会不同,
            # 因此可以用指数加权平均的方法计算当前batch的mean和var,
            # 即把从第一个batch到当前batch的所有mean和var的指数加权平均作为当前batch的mean和var。
            # 在test阶段, 就用最后一个batch的mean和var进行测试, 而不是采用test时的fc_mean和fc_var。
            # 将上面最后一步替换为如下代码, 就是计算当前batch的mean和var的方法。
            """
            # 创建指数加权平均模型
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            # 对fc_mean和fc_var应用指数加权平均
            ema_apply_op = ema.apply([fc_mean, fc_var])
            # mean和var的更新函数
            def mean_var_with_update():
                # tf.control_dependencies用来确保ema_apply_op会先于with中的操作被执行
                with tf.control_dependencies([ema_apply_op]):
                    # 取出当前batch的fc_mean和fc_var的指数加权平均
                    return ema.average(fc_mean), ema.average(fc_var)
            # 这一步用来确定是在train阶段还是在test阶段, 进而决定是否更新mean和var
            # 如果是在train阶段, 则进行更新, 否则直接采用上一次的结果
            # on_train是传入的参数, 值为True/False, 用来表明是否在train阶段
            # 通常可以把on_train定义成全局变量
            # 如果是True, 则调用mean_var_with_update更新mean和var
            # 如果是False, 返回之前fc_mean和fc_var的Moving Average
            mean, var = tf.cond(on_train,
                                mean_var_with_update,
                                lambda: (ema.average(fc_mean),
                                         ema.average(fc_var))
                                )
            Wx_plus_b = tf.nn.batch_normalization(
                Wx_plus_b, mean, var, shift, scale, epsilon)
            """

        if activator is None:
            outputs = Wx_plus_b
        else:
            outputs = activator(Wx_plus_b)
        return outputs

    # 固定随机数, 使得产生的结果相同
    fix_seed(1)

    # 对输入的特征进行批标准化
    if norm:
        fc_mean, fc_var = tf.nn.moments(x, axes=[0])
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        x = tf.nn.batch_normalization(
            x, fc_mean, fc_var, shift, scale, epsilon)

    # 记录每层的input
    layers_inputs = [x]

    for i in range(LAYER_COUNT):
        layer_input = layers_inputs[i]
        in_size = layer_input.get_shape()[1].value

        outputs = add_layer(
            layer_input,
            in_size,
            N_HIDDEN_UNITS,
            norm,
            activator=ACTIVATOR
        )

        layers_inputs.append(outputs)

    # 建立最后一层layer, 将输出特征转化为1维的
    prediction = add_layer(layers_inputs[-1], 30, 1, norm, activator=None)

    cost = tf.reduce_mean(tf.reduce_sum(
        tf.square(y - prediction), axis=1))
    train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train, cost, layers_inputs]


# 创建数据集
x_data = np.linspace(-7, 10, 500)[:, np.newaxis]  # 变成列向量
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise


# 创建用于接收输入的变量
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 无批标准化的网络
train, cost, layers_inputs = build_net(x, y, norm=False)
# 有批标准化的网络
train_norm, cost_norm, layers_inputs_norm = build_net(
    x, y, norm=True)


# 开始训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 记录两种网络的cost变化
cost_list = []
cost_list_norm = []
record_step = 5

plt.ion()  # 打开交互模式
# 创建一个figure, 用于显示输入分布
plt.figure(figsize=(7, 3))

for i in range(251):
    if i % 50 == 0:
        # 每50步显示每层的输入分布
        all_inputs, all_inputs_norm = sess.run(
            [layers_inputs, layers_inputs_norm], feed_dict={x: x_data, y: y_data})
        plot_inputs(all_inputs, all_inputs_norm)

    sess.run(train, feed_dict={x: x_data, y: y_data})
    sess.run(train_norm, feed_dict={x: x_data, y: y_data})

    if i % record_step == 0:
        # 每record_step步记录cost
        cost_list.append(sess.run(cost, feed_dict={x: x_data, y: y_data}))
        cost_list_norm.append(
            sess.run(cost_norm, feed_dict={x: x_data, y: y_data}))

plt.ioff()  # 关闭交互模式
# 创建一个figure, 用于显示损失曲线
plt.figure()
# 显示无批标准化的网络的损失曲线
plt.plot(np.arange(len(cost_list))*record_step,
         np.array(cost_list), label='no BN')
# 显示有批标准化的网络的损失曲线
plt.plot(np.arange(len(cost_list_norm))*record_step,
         np.array(cost_list_norm), label='BN')
plt.legend()  # 显示图例
plt.show()
