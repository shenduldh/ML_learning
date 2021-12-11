import os
# 屏蔽tensorflow输出的log
# 设置1屏蔽一般信息，2屏蔽一般和警告，3屏蔽所有输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
import numpy as np


# tensorflow.compat.v1表示使用1.0版本的tensorflow
tf.disable_v2_behavior()  # 禁用2.0版本的特性 (即时运行)


#######################
## example1：预测方程 ##
######################

# 初始化训练集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.5+0.3

### construct the model start ###
# 创建模型参数变量, tf.Variable的参数指的是该变量的初始化方法
Weights = tf.Variable(tf.random_uniform([1], -1.0, 2.0))
biases = tf.Variable(tf.zeros([1]))

y_prediction = Weights*x_data+biases  # 预测值
loss = tf.reduce_mean(tf.square(y_prediction-y_data))  # 损失
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 梯度下降

# 用于初始化参数的初始化器
init = tf.global_variables_initializer()
### construct the model end ###

sess = tf.Session()  # 建立会话
sess.run(init)  # 启动初始化器

print('********** example1 **********')
for step in range(401):
    sess.run(train)  # 启动训练器进行模型的训练
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

sess.close()


##########################
## example2：求函数最小值 ##
#########################

# 清除在上面建立好的计算图 (重置计算图)
tf.reset_default_graph()

W = tf.Variable(0, dtype=tf.float32)
# 创建占位符，使得可以在训练时再传递具体的数值，特别用于MiniBatch
X = tf.placeholder(tf.float32, [3, 1])
loss = tf.add(tf.add(X[0, 0]*W**2, tf.multiply(X[1, 0], W)), X[2, 0])
# 等同于 loss = X[0, 0]*W**2+X[1, 0]*W+X[2, 0]
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

# 使用with来打开会话，这样就不用手动调用sess.close()了
with tf.Session() as sess:
    sess.run(init)
    print('********** example2 **********')
    for step in range(401):
        # 启动训练器进行模型的训练
        # feed_dict用于传递具体的数值给占位符X
        sess.run(train, feed_dict={
            X: np.array([[1.], [-10.], [25.]])
        })
        if step % 20 == 0:
            print(step, sess.run(W))
