from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

tf.disable_v2_behavior()


# 任务: 迁移vgg16模型, 重建其全连接层 (用来预测图片中猫或老虎的长度)
# vgg16已经知道如何区分猫和老虎, 我们要做的就是将其迁移到我们的网络中,
# 在它识别图片中猫和老虎的基础上, 预测图片中猫或老虎的长度


vgg16_npy_path = './for_transfer_learning/vgg16.npy'
restore_path = './for_transfer_learning/model/transfer_learn'


def download():
    categories = ['tiger', 'kittycat']
    for category in categories:
        os.makedirs('./for_transfer_learning/data/%s' %
                    category, exist_ok=True)
        with open('./for_transfer_learning/imagenet_%s.txt' % category, 'r') as file:
            urls = file.readlines()
            n_urls = len(urls)
            for i, url in enumerate(urls):
                try:
                    urlretrieve(url.strip(), './for_transfer_learning/data/%s/%s' %
                                (category, url.strip().split('/')[-1]))
                    print('%s %i/%i' % (category, i, n_urls))
                except:
                    print('%s %i/%i' % (category, i, n_urls), 'no image')


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))[
        None, :, :, :]  # shape [1, 224, 224, 3]
    return resized_img


def load_data():
    imgs = {'tiger': [], 'kittycat': []}
    for k in imgs.keys():
        dir = './for_transfer_learning/data/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            imgs[k].append(resized_img)  # [1, height, width, depth] * n
            if len(imgs[k]) == 140:      # only use 140 imgs
                break
    # 生成虚假的体长标签
    tigers_y = np.maximum(20, np.random.randn(
        len(imgs['tiger']), 1) * 30 + 100)
    cat_y = np.maximum(10, np.random.randn(len(imgs['kittycat']), 1) * 8 + 40)
    return imgs['tiger'], imgs['kittycat'], tigers_y, cat_y


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_path=None):
        try:
            # 读取vgg16模型的已训练好的参数, 返回的是一个包含各层参数的字典
            self.params_dict = np.load(
                vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM')
            print(
                'Or from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, [None, 1])

        # 将图片数据的RGB通道转化为BGR通道, 因为vgg16模型采用的是BGR通道
        red, green, blue = tf.split(
            axis=3, num_or_size_splits=3, value=self.x * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # 下面是vgg16模型中已经训练好的网络层 (和原模型保持一致)
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2)

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2)

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3)

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3)

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3)

        # 将vgg16模型中的全连接层替换为服务于我们任务 (识别图中猫或老虎的长度) 的全连接层
        # 将vgg16模型中已训练好的部分的参数固定下来, 只训练我们添加的全连接层的参数
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu)
        self.out = tf.layers.dense(self.fc6, 1)

        self.sess = tf.Session()
        if restore_path:
            # 如果是应用阶段, 则直接读取已训练好的全连接层的参数
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
        else:
            # 如果是训练阶段, 则将全连接层的参数进行初始化
            self.loss = tf.losses.mean_squared_error(
                labels=self.y, predictions=self.out)
            self.train_op = tf.train.RMSPropOptimizer(
                0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, inputs):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def conv_layer(self, inputs, name):
        # VGG16中CNN部分的卷积核的参数都采用恒量, 而不是变量
        # 因此在训练阶段, 这部分的参数固定不变
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(inputs, self.params_dict[name][0], [
                                1, 1, 1, 1], padding='SAME')
            out = tf.nn.relu(tf.nn.bias_add(conv, self.params_dict[name][1]))
            return out

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {
                                self.x: x, self.y: y})
        return loss

    def predict(self, paths):
        fig, axis = plt.subplots(1, 2)
        for i, path in enumerate(paths):
            x = load_img(path)
            length = self.sess.run(self.out, {self.x: x})
            axis[i].imshow(x[0])
            axis[i].set_title('Len: %.1f cm' % length)
            axis[i].set_xticks(())
            axis[i].set_yticks(())
        plt.show()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    tigers_x, cats_x, tigers_y, cats_y = load_data()

    显示猫和老虎图片中它们体长的分布
    plt.hist(tigers_y, bins=20, label='Tigers')
    plt.hist(cats_y, bins=10, label='Cats')
    plt.legend()
    plt.xlabel('length')
    plt.show()

    xs = np.concatenate(tigers_x + cats_x, axis=0)
    ys = np.concatenate((tigers_y, cats_y), axis=0)

    vgg = Vgg16(vgg16_npy_path=vgg16_npy_path)
    print('Net built!')
    for i in range(100):
        b_idx = np.random.randint(0, len(xs), 6)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        print(i, 'train loss: ', train_loss)

    # 保存训练好的参数
    vgg.save(restore_path)


def eval():
    vgg = Vgg16(vgg16_npy_path=vgg16_npy_path,
                restore_path=restore_path)
    vgg.predict(
        ['./for_transfer_learning/data/kittycat/test.jpg', './for_transfer_learning/data/tiger/test.jpg'])


if __name__ == '__main__':
    # download()
    train()
    # eval()
