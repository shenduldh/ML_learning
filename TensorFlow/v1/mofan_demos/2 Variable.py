import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 例子1: 计数器
# 创建变量, 初始值为0, 别名为counter（这个别名用于tensorboard的可视化命名）
state = tf.Variable(0, name='counter')
one = tf.constant(1)  # 创建常量, 值为1
new_value = tf.add(state, one)  # 做加法
update = tf.assign(state, new_value)  # 将新的值赋给变量state
init = tf.global_variables_initializer()  # 用于初始变量的初始化器

with tf.Session() as sess:
    sess.run(init)  # 启动初始化器来初始变量
    for _ in range(10):
        sess.run(update)  # 启动更新器
        result = sess.run(state)  # 取出变量state的值
        print(result)


# 例子2: 用变量定义函数 loss=(y'-y)^2
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')
# 定义一个tensorflow变量, 这个变量就表示了上面的loss函数
loss = tf.Variable((y - y_hat)**2, name='loss')

init = tf.global_variables_initializer()

with tf.Session() as session:
    # session.run之前的都是设想, session.run时才是执行那些设想。
    # 就像建一座大厦, session.run之前都是在设计, session.run时才是按设计图动工。
    session.run(init)
    print(session.run(loss))
