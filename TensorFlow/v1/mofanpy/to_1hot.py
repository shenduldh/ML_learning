import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def to_1hot(labels, class_count):
    C = tf.constant(class_count, name='C')

    # 使用tf.one_hot函数构建转换操作
    to_1hot_op = tf.one_hot(indices=labels, depth=C, axis=0)

    sess = tf.Session()
    one_hot_encoding = sess.run(to_1hot_op)
    sess.close()

    return one_hot_encoding
