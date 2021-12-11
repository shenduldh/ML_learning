import tensorflow as tf
from tensorflow import keras
from utils import process_w2v_data
from visual import show_w2v_word_embedding


corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]


# 此处 skipgram 和 cbow 的算法流程基本一样,
# 因为 cbow 是用一个上下文向量预测一个中心词向量, 是 1 vs 1 的过程,
# 而此处 skipgram 是用一个中心词向量预测一个上下文词向量, 也是 1 vs 1 的过程,
# 因此它们的算法流程基本一样。那么为什么这里的 skipgram 不是一对多的？
# 为了达到一对一的效果, 在数据预处理时进行了特别操作, 原本应该是这样:
#   x: 莫 -> y: [我, 爱, 烦, Python]
# 现在我们把这个一次预测的过程分解成四次预测的过程来进行 (一个样本分成了四个样本):
#   x: 莫 -> y: 我
#   x: 莫 -> y: 爱
#   x: 莫 -> y: 烦
#   x: 莫 -> y: Python
# 所以表面上是一对一的过程, 实际上还是一对多的, 只不过把这个一对多的过程拆分成了多个一对一的形式。
# 经过这样一个处理, skipgram 和 cbow 的算法流程就基本一样了, 只是要在数据处理时做一下变化。


class SkipGram(keras.Model):
    def __init__(self, v_dim, emb_dim):
        super().__init__()

        self.v_dim = v_dim

        # 创建用于获取所有词的 embedding 的向量表, 对应 CBOW 的第一个 hidden layer
        self.embeddings = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),
        )

        # 采用负采样代替原始 CBOW 的第二个 hidden layer
        self.nce_w = self.add_weight(
            name="nce_w", shape=[v_dim, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0., 0.1))  # [n_vocab, emb_dim]
        self.nce_b = self.add_weight(
            name="nce_b", shape=(v_dim,),
            initializer=keras.initializers.Constant(0.1))  # [n_vocab, ]

        self.opt = keras.optimizers.Adam(0.01)

    def call(self, x, training=None, mask=None):
        # x.shape = [n, ]
        # 获取中心词的 embedding
        central_word_embedding = self.embeddings(x)  # [n, emb_dim]
        return central_word_embedding

    # negative sampling: take one positive label and num_sampled negative labels to compute the loss
    # in order to reduce the computation of full softmax
    def loss(self, x, y, training=None):
        central_word_embedding = self.call(x, training)
        return tf.reduce_mean(tf.nn.nce_loss(
            weights=self.nce_w,
            biases=self.nce_b,
            labels=tf.expand_dims(y, axis=1),
            inputs=central_word_embedding,
            num_sampled=5, num_classes=self.v_dim
        ))

    def step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.loss(x, y, True)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


def train(model, data):
    for t in range(2500):
        bx, by = data.sample(8)
        loss = model.step(bx, by)
        if t % 200 == 0:
            print("step: {} | loss: {}".format(t, loss))


if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method="skip_gram")
    m = SkipGram(d.num_word, 2)
    train(m, d)

    # plot
    show_w2v_word_embedding(m, d, "./skipgram.png")


# 注意, word2vec 无法处理一词多义的情况。
# 比如在"我是阳光男孩"和"今天阳光明媚"中, word2vec认为"阳光"都是一样的含义。
# 解决办法: 如果能考虑到句子上下文的信息, 那么这个词向量就能表达词语在不同句子中不同的含义了。
