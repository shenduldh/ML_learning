import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 让tensorflow使用CPU

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


input1 = tf.placeholder(tf.float32) # 占位符，在运行时才赋给其具体的值
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # 通过feed_dict给占位符赋值
    result = sess.run(output, feed_dict={
        input1: [1.],
        input2: [2.]
    })
    print(result)
