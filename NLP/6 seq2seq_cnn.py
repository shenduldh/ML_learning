import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils
import tensorflow_addons as tfa


# 任务: 将中文的顺序日期，转成英文的逆序日期，数据区间是20世纪后期到21世纪前期。
# 结构: 此模型的结构与 seq2seq 相似, 也是一个 encoder 和一个 decoder,
#      encoder 采用 cnn 的方式去提取句子的句向量,
#      而 decoder 是和 seq2seq 模型完全一样的过程, 均是采用 LSTM 进行处理。


# encoder 的理解:
# 对于一句话, 由每个词的词向量组成的矩阵 embeddings 就和只有 1 个 chanel 的图像数据一样,
# 因此完全可以将图像的 cnn 模型直接应用到文本处理中。具体过程如下:
#   ① 首先为 embeddings 增加 chanel 维度, 使其和图像数据完全一样;
#   ② 然后在 embeddings 上分别应用三个卷积层, 提取三个 feature map;
#   ③ 对三个 feature map 分别应用三个池化层, 使它们归一化到同一 dimension;
#   ④ 将归一化的三个 feature map 合并成句向量, 作为后续 decoder 的输入。
# 三个卷积层的卷积核大小都不一样, 目的是为了从不同的视角去获取对句子的理解,
# 池化层的作用是为了可以将这些理解能够合并起来, 统一作为句子的句向量。


class CNNTranslation(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        super().__init__()

        self.units = units

        # encoder
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        # 创建三个卷积层
        self.conv2ds = [
            keras.layers.Conv2D(16, (n, emb_dim),
                                padding="valid",
                                activation=keras.activations.relu)
            for n in range(2, 5)]
        # 创建三个池化层
        self.max_pools = [keras.layers.MaxPool2D((n, 1)) for n in [7, 6, 5]]
        # 创建一个全连接层
        self.encoder = keras.layers.Dense(
            units, activation=keras.activations.relu)

        # decoder
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,  # [dec_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.decoder_cell = keras.layers.LSTMCell(units=units)
        decoder_dense = keras.layers.Dense(dec_v_dim)
        # train decoder
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),  # sampler for train
            output_layer=decoder_dense
        )
        # predict decoder
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),  # sampler for predict
            output_layer=decoder_dense
        )

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.opt = keras.optimizers.Adam(0.01)
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x):
        # embedded.shape = [n, step, emb]
        embedded = self.enc_embeddings(x)

        # o.shape = [n, step=8, emb=16, 1], 相当于 8×16×1 的图片
        o = tf.expand_dims(embedded, axis=3)  # 在 axis=3 上增加一个 chanel 维度

        # co.shape = [ [n, 7, 1, 16], [n, 6, 1, 16], [n, 5, 1, 16] ]
        # 用三个卷积层分别产生三个 feature map
        co = [conv2d(o) for conv2d in self.conv2ds]

        # co.shape = [n, 1, 1, 16] * 3
        # 用三个池化层分别处理上面产生的三个 feature map,
        # 使它们在 axis=1 上的长度变成 1
        co = [self.max_pools[i](co[i]) for i in range(len(co))]

        # 下面通过展开、合并、全连接处理将上面三个 feature map 合并在一起,
        # 形成对句子的理解, 即句向量
        co = [tf.squeeze(c, axis=[1, 2]) for c in co]  # [n, 16] * 3
        o = tf.concat(co, axis=1)  # [n, 16*3]
        h = self.encoder(o)  # [n, units]

        # decoder 的 LSTM 的初始状态需要两个输入, 因此将结果复制一份,
        # 返回相同的输出作为 decoder 的 LSTM 的输入
        return [h, h]

    def inference(self, x):
        s = self.encode(x)
        done, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s,
        )
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(
                time=l, inputs=i, state=s, training=False)
            pred_id[:, l] = o.sample_id
        return pred_id

    def train_logits(self, x, y, seq_len):
        s = self.encode(x)
        dec_in = y[:, :-1]  # ignore <EOS>
        dec_emb_in = self.dec_embeddings(dec_in)
        o, _, _ = self.decoder_train(dec_emb_in, s, sequence_length=seq_len)
        logits = o.rnn_output
        return logits

    def step(self, x, y, seq_len):
        with tf.GradientTape() as tape:
            logits = self.train_logits(x, y, seq_len)
            dec_out = y[:, 1:]  # ignore <GO>
            loss = self.cross_entropy(dec_out, logits)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


def train():
    # get and process data
    data = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ",
          data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = CNNTranslation(
        data.num_word, data.num_word, emb_dim=16, units=32,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    # train
    for t in range(1500):
        bx, by, decoder_len = data.sample(32)
        loss = model.step(bx, by, decoder_len)
        if t % 70 == 0:
            target = data.idx2str(by[0, 1:-1])
            pred = model.inference(bx[0:1])
            res = data.idx2str(pred[0])
            src = data.idx2str(bx[0])
            print(
                "step:", t,
                "| loss:", loss,
                "| input:", src,
                "| target:", target,
                "| inference:", res,
            )


if __name__ == "__main__":
    train()
