import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def add_layer(inputs, in_size, out_size, n_layer, activator=None):
    layer_name = 'layer'+str(n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal(
                [in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights)+biases
        if activator is None:
            outputs = Wx_plus_b
        else:
            outputs = activator(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs', outputs)
        return outputs


###############
## 创建数据集 ##
##############
x_data = np.linspace(-1, 1, num=300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


#############
## 构建模型 ##
############
# with tf.name_scope('inputs'): 用于创建一个命名空间, with内的组件都会归并到这个空间内
# NOTE:
# tensorboard以每个组件 (运算函数、变量、常量等) 为单元, 每个组件都可以用name来重命名
# 如果不使用命名空间, 则创建的图就是以每个组件为一个节点
# 若使用命名空间来组织多个组件, 则这多个组件都会被认为是一个节点
with tf.name_scope('inputs'):
    # name='x_input': 用于取别名，不然在tensorboard中会显示默认名，即'placeholder'
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

layer_one = add_layer(xs, 1, 10, n_layer=1, activator=tf.nn.relu)
prediction = add_layer(layer_one, 10, 1, n_layer=2, activator=None)

with tf.name_scope('loss'):
    # 也可以给每一个运算函数取别名, 在tensorboard中就会用别名代替这些函数的名字
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys), axis=1))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


################
## 将图写入文件 ##
################
sess = tf.Session()
# 将上面创建的图作为文件写入logs目录,
# 然后在命令行使用tensorboard --logdir=logs命令,
# 就可以在浏览器打开tensorboard来查看该文件中的图
writer = tf.summary.FileWriter("logs", sess.graph)
# NOTE:
# tf.summary: 用于记录训练过程中数据的变化
# tf.summary.histogram(tagName, data): 用于将指定数据加入到histograms中
# tf.summary.scalar(tagName, data): 用于将指定数据加入到scalars中
# tf.summary.merge_all(): 用于将添加到summary的所有数据合并到一起, 以便后续训练时记录数据的变化
mergedSummary = tf.summary.merge_all()
sess.run(init)

for step in range(1001):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if step % 50 == 0:
        # 获取此时summary中记录的所有数据（存储在mergedSummary中）
        result = sess.run(mergedSummary, feed_dict={xs: x_data, ys: y_data})
        # 将数据记录到文件中
        writer.add_summary(result, step)
