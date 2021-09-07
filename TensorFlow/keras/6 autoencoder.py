import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np
np.random.seed(1337)  # for reproducibility


# 任务: 压缩和解压图片
# 目的: 学习如何使用 Model 这个函数来创建模型


(x_train, _), (x_test, y_test) = mnist.load_data()
# x_train: shape(60000, 28, 28)
# x_test: shape(10000, 28, 28)
# y_test: shape(10000, )

# preprocess data
x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
# x_train: shape(60000, 28×28)
# x_test: shape(10000, 28×28)


# in order to plot in a 2D figure
encoding_dim = 2


# this is our input placeholder
input_img = Input(shape=(784,))  # 说明单个样本的样子即可

# encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_outputs = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_outputs)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoder_outputs = Dense(784, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(inputs=input_img, outputs=decoder_outputs)

# construct the encoder model for plotting
encoder = Model(inputs=input_img, outputs=encoder_outputs)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')


# train
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True)

# plot
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()
