import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils
import tensorflow_addons as tfa


# 任务: 将中文的顺序日期，转成英文的逆序日期，数据区间是20世纪后期到21世纪前期。
# 结构: RNN ENCODER–DECODER
#      ① encoder LSTM: 输入为一句话中各个词的词向量组成的序列;
#                       输出为最后一个时间步的状态和输出 (作为句向量)
#      ② decoder LSTM: 初始状态为 encoder 输出的句向量, 其余时间步的输入为上一时间步预测的词向量;
#                       输出为每一个时间步预测的词向量组成的一句话


# 在进行 seq2seq 前, 先为所有词创建了索引和词向量,
# keras.layers.Embedding 根据词的索引获取其词向量。
# 也就是说, seq2seq 是将一组词转换成另一组词,
# 而且这些词都是事先准备好的, 不是 seq2seq 模型在转换时凭空创造的。


# 此处 seq2seq 模型在训练过程和预测过程 (inference) 的 decoder 是不同的 (但 encoder 是一样的),
# 具体表现为: 用于 decode 的 LSTM 的每一时间步 (除了初始状态) 的输入不同
#   ① 在训练过程中, LSTM 每一时间步的输入都为正确标签的词向量,
#      这样做的目的是使得不管在训练时有没有预测错, 下一步的输入都是正确的
#   ② 在 inference 中, LSTM 每一时间步的输入就是上一时间步预测的词向量


# 进一步优化: 可以将 GreedyEmbeddingSampler 更改为 Beam search。
# decoder 在每一时间步进行预测时都会输出多个预测结果 (概率分布)。
# GreedyEmbeddingSampler 是指每次预测的结果都只采用最优的那个预测。
# Beam search 是指每次预测都记录最优的两个预测, 然后沿着这两个预测分别继续预测, 不断分叉, 最后形成一颗树,
# 该树的每一条路径就代表了一个预测的序列, 选其中最优的那个作为最终翻译的目标序列。
# 此处的 seq2seq 采用的是 GreedyEmbeddingSampler 的方法。


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units

        # encoder
        # 用于获取各个词的词向量
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        # 根据一句话中各个词的词向量, 得到这句话的句向量
        self.encoder = keras.layers.LSTM(
            units=units, return_sequences=True, return_state=True)

        # decoder
        # 用于获取目标序列中各个词的词向量
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,  # [dec_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        # 用于 decoder 的 LSTM
        self.decoder_cell = keras.layers.LSTMCell(units=units)
        decoder_dense = keras.layers.Dense(dec_v_dim)

        # train decoder
        # 将上面定义的 LSTM 包装成专门用于 seq2seq 训练的 decoder,
        # 因为训练时和预测时用的 decoder 有区别
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            # The tfa.seq2seq.Sampler instance passed as argument
            # is responsible to sample from the output distribution
            # and produce the input for the next decoding step
            sampler=tfa.seq2seq.sampler.TrainingSampler(),  # sampler for train
            output_layer=decoder_dense
        )

        # predict decoder
        # 将上面定义的 LSTM 包装成专门用于 seq2seq 预测的 decoder,
        # 因为训练时和预测时用的 decoder 有区别
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),  # sampler for predict
            output_layer=decoder_dense
        )

        # 下面分别是损失函数、优化器的定义
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.opt = keras.optimizers.Adam(0.01)

        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    # encoder 的整个过程
    def encode(self, x):
        # 获取一句话中各个词的词向量
        embedded = self.enc_embeddings(x)
        # 初始化 encoder 中 LSTM 的初始状态
        init_s = [tf.zeros((x.shape[0], self.units)),
                  tf.zeros((x.shape[0], self.units))]
        # 将一句话中各个词的词向量作为序列输入到 LSTM 中
        # 将最后一个时间步的输出 h 和状态 c 作为对这句话的理解 (句向量)
        o, h, c = self.encoder(embedded, initial_state=init_s)
        return [h, c]

    # 预测步骤
    def inference(self, x):
        s = self.encode(x)
        done, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s,
        )

        # pred_id 用于存放每一个需要翻译的序列的翻译结果的索引,
        # 每一行就是一个序列被翻译后的结果的索引列表
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(
                time=l, inputs=i, state=s, training=False)
            pred_id[:, l] = o.sample_id
        return pred_id

    # 训练步骤
    def train_logits(self, x, y, seq_len):
        s = self.encode(x)
        dec_in = y[:, :-1]  # ignore <EOS>
        # 获取所有正确标签的词向量 dec_emb_in
        dec_emb_in = self.dec_embeddings(dec_in)
        # 用 dec_emb_in 作为每个时间步 (除第一个时间步外) 的输入
        o, _, _ = self.decoder_train(dec_emb_in, s, sequence_length=seq_len)
        logits = o.rnn_output
        return logits

    # 训练步骤
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
    data = utils.DateData(4000)  # 由语料库、中英文日期 (字符串形式和索引形式)构成
    print("Chinese time order: yy/mm/dd ",
          data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = Seq2Seq(
        data.num_word, data.num_word, emb_dim=16, units=32,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    # train
    for t in range(1500):
        bx, by, decoder_len = data.sample(32)
        loss = model.step(bx, by, decoder_len)
        if t % 70 == 0:
            target = data.idx2str(by[0, 1:-1])
            # 每次只翻译一个序列, 因此也只是返回一个翻译后的目标序列, 由索引组成
            pred = model.inference(bx[0:1])
            # 将由索引组成的目标序列转换成字符串形式
            res = data.idx2str(pred[0])
            # 将由索引组成的源序列转换成字符串形式
            src = data.idx2str(bx[0])
            print("step:", t,
                  "| loss:", loss,
                  "| input:", src,
                  "| target:", target,
                  "| inference:", res,)


if __name__ == "__main__":
    train()
