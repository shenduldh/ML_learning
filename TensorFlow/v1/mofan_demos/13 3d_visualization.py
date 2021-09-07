import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tf.disable_v2_behavior()

learning_rate = 0.1
real_params = [3, 4]
init_params = [
    [1, 2],
    [3, 4],
    [5, 6]
][0]


def target_func(w1, w2): return w1*real_x+w2


real_x = np.linspace(-1, 1, 200, dtype=np.float32)
noise = np.random.randn(200)/10
real_y = target_func(*init_params)+noise

w1, w2 = [tf.Variable(p, dtype=tf.float32) for p in init_params]
prediction = target_func(w1, w2)
lost = tf.reduce_mean(tf.square(prediction-real_y))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(lost)


w1_list, w2_list, lost_list = [], [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(500):
        w1_, w2_, lost_, result = sess.run([w1, w2, lost, prediction])
        w1_list.append(w1_)
        w2_list.append(w2_)
        lost_list.append(lost_)
        sess.run(train)


# visualization codes
print('w1=', w1_, 'w2=', w2_)
plt.figure(1)
plt.scatter(real_x, real_y, c='b')
plt.plot(real_x, result, 'r-', lw=2)

# 3D cost figure
fig = plt.figure(2)
ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30),
                       np.linspace(-2, 7, 30))
cost3D = np.array([np.mean(np.square(target_func(w1_, w2_) - real_y))
                  for w1_, w2_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1,
                cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(w1_list[0], w2_list[0], zs=lost_list[0],
           s=300, c='r')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.plot(w1_list, w2_list, zs=lost_list, zdir='z',
        c='r', lw=3)
plt.show()
