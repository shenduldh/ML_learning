import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils
import tensorflow_addons as tfa
import pickle


# 任务: 将中文的顺序日期，转成英文的逆序日期，数据区间是20世纪后期到21世纪前期。
# 采用 LuongAttention 作为注意力机制。


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, attention_layer_size, max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units

        # encoder: 和原始 Seq2seq 模型中的 encoder 一样
        self.enc_embeddings = keras.layers.Embedding(
            # [enc_n_vocab, emb_dim]
            input_dim=enc_v_dim, output_dim=emb_dim,
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.encoder = keras.layers.LSTM(
            units=units, return_sequences=True, return_state=True)

        # decoder: 和原始 Seq2seq 模型中的 decoder 不同的是增加了 attention layer
        # 用这个 attention layer 去包裹原来的 LSTM cell, 使其具有 attention 的功能
        self.attention = tfa.seq2seq.LuongAttention(
            units, memory=None, memory_sequence_length=None)
        self.decoder_cell = tfa.seq2seq.AttentionWrapper(
            cell=keras.layers.LSTMCell(units=units),
            attention_mechanism=self.attention,
            attention_layer_size=attention_layer_size,
            alignment_history=True,  # for attention visualization
        )

        self.dec_embeddings = keras.layers.Embedding(
            # [dec_n_vocab, emb_dim]
            input_dim=dec_v_dim, output_dim=emb_dim,
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        decoder_dense = keras.layers.Dense(dec_v_dim)  # output layer

        # train decoder
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),  # sampler for train
            output_layer=decoder_dense
        )
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.opt = keras.optimizers.Adam(0.05, clipnorm=5.0)

        # predict decoder
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),  # sampler for predict
            output_layer=decoder_dense
        )

        # prediction restriction
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x):
        o = self.enc_embeddings(x)
        init_s = [tf.zeros((x.shape[0], self.units)),
                  tf.zeros((x.shape[0], self.units))]
        o, h, c = self.encoder(o, initial_state=init_s)
        return o, h, c

    def set_attention(self, x):
        o, h, c = self.encode(x)
        # encoder output for attention to focus
        # 将 encoder 每一时刻的信息缓存到 decoder 上, 以便后续计算动态语义向量
        self.attention.setup_memory(o)
        # wrap state by attention wrapper
        # 用 encode 最后时刻的状态和输出来初始化 decoder 的状态
        s = self.decoder_cell.get_initial_state(
            batch_size=x.shape[0], dtype=tf.float32).clone(cell_state=[h, c])
        return s

    def inference(self, x, return_align=False):
        s = self.set_attention(x)
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
        if return_align:
            return np.transpose(s.alignment_history.stack().numpy(), (1, 0, 2))
        else:
            s.alignment_history.mark_used()  # otherwise gives warning
            return pred_id

    def train_logits(self, x, y, seq_len):
        s = self.set_attention(x)
        dec_in = y[:, :-1]   # ignore <EOS>
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
    data = utils.DateData(2000)
    print("Chinese time order: yy/mm/dd ",
          data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = Seq2Seq(
        data.num_word, data.num_word, emb_dim=12, units=14, attention_layer_size=16,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    # training
    for t in range(1000):
        bx, by, decoder_len = data.sample(64)
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

    pkl_data = {"i2v": data.i2v, "x": data.x[:6], "y": data.y[:6], "align": model.inference(
        data.x[:6], return_align=True)}

    with open("./attention_align.pkl", "wb") as f:
        pickle.dump(pkl_data, f)


if __name__ == "__main__":
    train()
