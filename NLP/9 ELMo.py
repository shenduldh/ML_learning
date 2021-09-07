from tensorflow import keras
import tensorflow as tf
import utils
import time
import os


class ELMo(keras.Model):
    def __init__(self, v_dim, emb_dim, units, n_layers, lr):
        super().__init__()
        self.n_layers = n_layers
        self.units = units

        # encoder
        self.word_embed = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.001),
            mask_zero=True,
            # mask_zero=True: 表示用于填充序列的值为 0,
            # 使得后续调用 compute_mask 时,
            # 可以根据值为 0 的元素位置得知 padding 部分,
            # 进而求得 mask 矩阵
        )

        # forward lstm
        # 从前向开始从上文预测下一个词
        self.fs = [keras.layers.LSTM(units, return_sequences=True)
                   for _ in range(n_layers)]
        # 用于转化前向预测得到的词向量为 n 维向量 (n 为词类别个数),
        # 进而计算前向预测的损失
        self.f_logits = keras.layers.Dense(v_dim)

        # backward lstm
        # 从反向开始从下文预测上一个词
        # go_backwards=True: 表明是逆向读取序列
        self.bs = [keras.layers.LSTM(units, return_sequences=True, go_backwards=True)
                   for _ in range(n_layers)]
        # 用于转化反向预测得到的词向量为 n 维向量 (n 为词类别个数),
        # 进而计算反向预测的损失
        self.b_logits = keras.layers.Dense(v_dim)

        self.cross_entropy1 = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.cross_entropy2 = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.opt = keras.optimizers.Adam(lr)

    def call(self, seqs):
        """
        0123    forward
        1234    forward predict
        1234    backward 
        0123    backward predict
        """
        embedded = self.word_embed(seqs)  # [n, step, dim]
        # 这里的 mask 矩阵表明了序列的真实长度,
        # 使得 LSTM 在预测时只会处理真正属于序列中的词,
        # 而不会去处理用于填充序列的无效词
        mask = self.word_embed.compute_mask(seqs)
        fxs, bxs = [embedded[:, :-1]], [embedded[:, 1:]]
        for fl, bl in zip(self.fs, self.bs):
            # 每次返回所有时间步的 hidden state (即输出),
            # 而不是 cell state (即记忆)
            fx = fl(fxs[-1], mask=mask[:, :-1],
                    initial_state=fl.get_initial_state(fxs[-1]))  # [n, step-1, dim]
            bx = bl(bxs[-1], mask=mask[:, 1:],
                    initial_state=bl.get_initial_state(bxs[-1]))  # [n, step-1, dim]
            # 将每一层的所有时间步的 hidden state 都保存起来
            fxs.append(fx)  # predict 1234
            bxs.append(tf.reverse(bx, axis=[1]))  # predict 0123
        return fxs, bxs

    def step(self, seqs):
        with tf.GradientTape() as tape:
            fxs, bxs = self.call(seqs)
            # 将正向预测和反向预测的词向量 (最后一层的 hidden state) 转化为 n 维向量,
            # 进而在计算 loss 时通过 softmax 转化为概率分布,
            # 概率最高的那个元素所代表的词就是预测的词, 然后与 label 比较计算出 loss。
            fo, bo = self.f_logits(fxs[-1]), self.b_logits(bxs[-1])
            # 正向预测的损失加上反向预测的损失即为最终的优化目标
            loss = (self.cross_entropy1(seqs[:, 1:], fo) +
                    self.cross_entropy2(seqs[:, :-1], bo))/2
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, (fo, bo)

    # 用于从 elmo 模型中得到指定序列中每个词的词向量
    def get_emb(self, seqs):
        fxs, bxs = self.call(seqs)
        xs = [
            # from word embedding
            tf.concat((fxs[0][:, 1:, :], bxs[0][:, :-1, :]), axis=2).numpy()
        ] + [
            # from sentence embedding
            tf.concat((f[:, :-1, :], b[:, 1:, :]), axis=2).numpy()
            for f, b in zip(fxs[1:], bxs[1:])
        ]
        for x in xs:
            print("layers shape=", x.shape)
        return xs


def train(model, data, step):
    t0 = time.time()
    for t in range(step):
        seqs = data.sample(BATCH_SIZE)
        loss, (fo, bo) = model.step(seqs)
        if t % 80 == 0:
            fp = fo[0].numpy().argmax(axis=1)
            bp = bo[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i]
                                       for i in seqs[0] if i != data.pad_id]),
                "\n| f_prd: ", " ".join([data.i2v[i]
                                         for i in fp if i != data.pad_id]),
                "\n| b_prd: ", " ".join([data.i2v[i]
                                         for i in bp if i != data.pad_id]),
            )
            t0 = t1
    os.makedirs("./visual/models/elmo", exist_ok=True)
    model.save_weights("./visual/models/elmo/model.ckpt")


def export_w2v(model, data):
    model.load_weights("./visual/models/elmo/model.ckpt")
    emb = model.get_emb(data.sample(4))
    print(emb)


if __name__ == "__main__":
    utils.set_soft_gpu(True)
    UNITS = 256
    N_LAYERS = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-3
    d = utils.MRPCSingle("./MRPC", rows=2000)
    print("num word: ", d.num_word)
    m = ELMo(d.num_word, emb_dim=UNITS, units=UNITS,
             n_layers=N_LAYERS, lr=LEARNING_RATE)
    train(m, d, 10000)
    export_w2v(m, d)
