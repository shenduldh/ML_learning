import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
b = tf.Variable([[1, 2, 3]], dtype=tf.float32)

initializer = tf.global_variables_initializer()

saver = tf.train.Saver()  # 创建保存器

with tf.Session() as sess:
    sess.run(initializer)
    # 将此次会话中创建的Varialbe保存到saved_variable/demo.ckpt中
    save_path = saver.save(sess, 'saved_variable/demo.ckpt')
    print('Save to path:'+save_path)
