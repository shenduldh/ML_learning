import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils
import time
import pickle
import os


MODEL_DIM = 32
MAX_LEN = 12
N_LAYER = 3
N_HEAD = 4
DROP_RATE = 0.1


# 用于计算多头注意力的 layer
class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = keras.layers.Dense(n_head * self.head_dim)
        self.wk = keras.layers.Dense(n_head * self.head_dim)
        self.wv = keras.layers.Dense(
            n_head * self.head_dim)  # [n, step, h*h_dim]

        self.o_dense = keras.layers.Dense(model_dim)
        self.o_drop = keras.layers.Dropout(rate=drop_rate)
        self.attention = None

    def call(self, q, k, v, mask, training):
        _q = self.wq(q)  # [n, q_step, h*h_dim]
        _k, _v = self.wk(k), self.wv(v)  # [n, step, h*h_dim]
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(
            _v)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(
            _q, _k, _v, mask)  # [n, q_step, h*dv]
        o = self.o_dense(context)  # [n, step, dim]
        o = self.o_drop(o, training=training)
        return o

    def split_heads(self, x):
        # [n, step, h, h_dim]
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [n, h, step, h_dim]

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / \
            (tf.math.sqrt(dk) + 1e-8)  # [n, h_dim, q_step, step]
        # here are padding mask and sequence mask
        if mask is not None:
            score += mask * -1e9
        # [n, h, q_step, step]
        self.attention = tf.nn.softmax(score, axis=-1)
        # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        context = tf.matmul(self.attention, v)
        # [n, q_step, h, dv]
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        # [n, q_step, h*dv]
        context = tf.reshape(context, (context.shape[0], context.shape[1], -1))
        return context


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        dff = model_dim * 4
        self.l = keras.layers.Dense(dff, activation=keras.activations.relu)
        self.o = keras.layers.Dense(model_dim)

    def call(self, x):
        o = self.l(x)
        o = self.o(o)
        return o  # [n, step, dim]


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(
            axis=-1) for _ in range(2)]  # only norm z-dim
        self.mh = MultiHead(n_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim)
        self.drop = keras.layers.Dropout(drop_rate)

    def call(self, xz, training, mask):
        # 注意力 layer
        attn = self.mh.call(xz, xz, xz, mask, training)  # [n, step, dim]
        # 残差+规范化 layer
        o1 = self.ln[0](attn + xz)
        # 前向网络 layer
        ffn = self.drop(self.ffn.call(o1), training)
        # 残差+规范化 layer
        o = self.ln[1](ffn + o1)  # [n, step, dim]
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate)
                   for _ in range(n_layer)]

    def call(self, xz, training, mask):
        # 多层 Encoder 的计算
        for l in self.ls:
            xz = l.call(xz, training, mask)
        return xz  # [n, step, dim]


class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(
            axis=-1) for _ in range(3)]  # only norm z-dim
        self.drop = keras.layers.Dropout(drop_rate)
        self.mh = [MultiHead(n_head, model_dim, drop_rate) for _ in range(2)]
        self.ffn = PositionWiseFFN(model_dim)

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        # decoder self attention
        attn = self.mh[0].call(yz, yz, yz, yz_look_ahead_mask, training)
        # 残差+规范化
        o1 = self.ln[0](attn + yz)
        # decoder_encoder attention
        attn = self.mh[1].call(o1, xz, xz, xz_pad_mask, training)
        # 残差+规范化
        o2 = self.ln[1](attn + o1)
        # 前向网络 layer
        ffn = self.drop(self.ffn.call(o2), training)
        # 残差+规范化
        o = self.ln[2](ffn + o2)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [DecoderLayer(n_head, model_dim, drop_rate)
                   for _ in range(n_layer)]

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        # 多层 Decoder 的计算
        for l in self.ls:
            yz = l.call(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, model_dim, n_vocab):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        pe = pos / np.power(10000, 2. * np.arange(model_dim)
                            [None, :] / model_dim)  # [max_len, dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = pe[None, :, :]  # [1, max_len, model_dim] for batch adding
        self.pe = tf.constant(pe, dtype=tf.float32)
        self.embeddings = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )

    def call(self, x):
        x_embed = self.embeddings(x) + self.pe  # [n, step, dim]
        return x_embed


class Transformer(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, drop_rate=0.1, padding_idx=0):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        self.embed = PositionEmbedding(max_len, model_dim, n_vocab)
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, model_dim, drop_rate, n_layer)
        self.o = keras.layers.Dense(n_vocab)

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(0.002)

    def call(self, x, y, training=None):
        x_embed, y_embed = self.embed(x), self.embed(y)
        pad_mask = self._pad_mask(x)
        encoded_z = self.encoder.call(x_embed, training, mask=pad_mask)
        # 用 ground true 一次生成所有词
        decoded_z = self.decoder.call(
            y_embed, encoded_z, training, yz_look_ahead_mask=self._look_ahead_mask(y), xz_pad_mask=pad_mask)
        o = self.o(decoded_z)
        return o

    def step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.call(x, y[:, :-1], training=True)
            pad_mask = tf.math.not_equal(y[:, 1:], self.padding_idx)
            loss = tf.reduce_mean(tf.boolean_mask(
                self.cross_entropy(y[:, 1:], logits), pad_mask))
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, logits

    def _pad_bool(self, seqs):
        return tf.math.equal(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        mask = tf.cast(self._pad_bool(seqs), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    def _look_ahead_mask(self, seqs):
        mask = 1 - \
            tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        mask = tf.where(self._pad_bool(
            seqs)[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask  # (step, step)

    def translate(self, src, v2i, i2v):
        src_pad = utils.pad_zero(src, self.max_len)
        tgt = utils.pad_zero(
            np.array([[v2i["<GO>"], ] for _ in range(len(src))]), self.max_len+1)
        tgti = 0
        x_embed = self.embed(src_pad)
        encoded_z = self.encoder.call(
            x_embed, False, mask=self._pad_mask(src_pad))
        # 用上一个预测生成的词去预测生成下一个词, 直到超出最大句子长度
        while True:
            y = tgt[:, :-1]
            y_embed = self.embed(y)
            decoded_z = self.decoder.call(
                y_embed, encoded_z, False, yz_look_ahead_mask=self._look_ahead_mask(y), xz_pad_mask=self._pad_mask(src_pad))
            # 将所有词都放在一起 decoder（即使有些词还没有被生成，还是空的）,
            # 在 decoder 结束后, 只取出当前所要预测生成的那一个即可
            logits = self.o(decoded_z)[:, tgti, :].numpy()
            idx = np.argmax(logits, axis=1)
            tgti += 1
            # 将预测得到的词的索引作为结果保存起来
            tgt[:, tgti] = idx
            if tgti >= self.max_len:
                break
        return ["".join([i2v[i] for i in tgt[j, 1:tgti]]) for j in range(len(src))]

    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.numpy() for l in self.encoder.ls],
            "decoder": {
                "mh1": [l.mh[0].attention.numpy() for l in self.decoder.ls],
                "mh2": [l.mh[1].attention.numpy() for l in self.decoder.ls],
            }}
        return attentions


def train(model, data, step):
    t0 = time.time()
    for t in range(step):
        bx, by, seq_len = data.sample(64)
        bx, by = utils.pad_zero(bx, max_len=MAX_LEN), utils.pad_zero(
            by, max_len=MAX_LEN + 1)
        loss, logits = model.step(bx, by)
        if t % 50 == 0:
            logits = logits[0].numpy()
            t1 = time.time()
            print(
                "step: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.4f" % loss.numpy(),
                "| target: ", "".join([data.i2v[i] for i in by[0, 1:10]]),
                "| inference: ", "".join(
                    [data.i2v[i] for i in np.argmax(logits, axis=1)[:10]]),
            )
            t0 = t1

    os.makedirs("./visual/models/transformer", exist_ok=True)
    model.save_weights("./visual/models/transformer/model.ckpt")
    os.makedirs("./visual/tmp", exist_ok=True)
    with open("./visual/tmp/transformer_v2i_i2v.pkl", "wb") as f:
        pickle.dump({"v2i": data.v2i, "i2v": data.i2v}, f)


def export_attention(model, data, name="transformer"):
    with open("./visual/tmp/transformer_v2i_i2v.pkl", "rb") as f:
        dic = pickle.load(f)
    model.load_weights("./visual/models/transformer/model.ckpt")
    bx, by, seq_len = data.sample(32)
    model.translate(bx, dic["v2i"], dic["i2v"])
    attn_data = {
        "src": [[data.i2v[i] for i in bx[j]] for j in range(len(bx))],
        "tgt": [[data.i2v[i] for i in by[j]] for j in range(len(by))],
        "attentions": model.attentions}
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(attn_data, f)


if __name__ == "__main__":
    utils.set_soft_gpu(True)
    d = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ",
          d.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", d.date_en[:3])
    print("vocabularies: ", d.vocab)
    print("x index sample: \n{}\n{}".format(d.idx2str(d.x[0]), d.x[0]),
          "\ny index sample: \n{}\n{}".format(d.idx2str(d.y[0]), d.y[0]))

    m = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_HEAD, d.num_word, DROP_RATE)
    train(m, d, step=800)
    export_attention(m, d)
