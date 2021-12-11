from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

for i in range(1000):
    origin_data, label = mnist.train.next_batch(1)
    label = np.argmax(label)

    data = np.matrix(origin_data).reshape(28, 28)
    data = data*255

    new_im = Image.fromarray(data.astype(np.uint8))
    # new_im.show()
    new_im.save('MNIST_IMAGE/mnist%i_%i.png' % (i, label), dpi=(17., 17.))

    print('%i/%i' % (i+1, 1000))
