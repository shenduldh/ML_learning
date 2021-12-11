import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32)
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32)

saver = tf.train.Saver()  # 创建保存器

with tf.Session() as sess:
    # 将saved_variable/demo.ckpt中的变量恢复到此处创建的变量中。
    # 这个过程就相当于初始化，但初始化的内容是事先保存好的，
    # 因此可以省去实际初始化的代码，用restore来代替。
    # 由于只是替代了初始化工作，因此框架这些还是要自己定义，
    # 而且要和所保存变量所在框架一致，这样才能恢复成功，
    # 因为restore是根据框架的样子来进行恢复的。
    saver.restore(sess, 'saved_variable/demo.ckpt')
    print("weights: ", sess.run(W))
    print("biases: ", sess.run(b))


# save和restore的作用:
# 1. 保存训练数据，使得下次可以继续训练
# 2. 保存已训练好的模型参数，方便用于实际应用
