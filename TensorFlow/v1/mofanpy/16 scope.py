import tensorflow as tf


# 使用name_scope仅仅可以给tf.Variable创建的变量产生前缀
# 使用variable_scope可以给tf.get_variable和tf.Variable创建的变量产生前缀
with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[
                           1], dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)        # var1:0
    print(sess.run(var1))   # [ 1.]
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]


# 不管是在name_scope还是在variable_scope中,
# tf.Variable创建的变量永远都是新的变量,
# 想要重用变量只能使用tf.get_variable指定相同变量的名字才行,
# 并且在重用变量前, 必须先调用scope.reuse_variables函数,
# 以说明下面的代码中我们需要重用变量。
# 重用变量一般使用在需要权值共享的模型当中, 比如RNN模型。
with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[
                           1], dtype=tf.float32, initializer=initializer)
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    scope.reuse_variables()
    var3_reuse = tf.get_variable(name='var3',)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [ 3.]
    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [ 4.]
    print(var4_reuse.name)      # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse))  # [ 4.]
    print(var3_reuse.name)      # a_variable_scope/var3:0
    print(sess.run(var3_reuse))  # [ 3.]
