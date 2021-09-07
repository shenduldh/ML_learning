import numpy as np
import tensorflow as tf
import utils
import time
import os
import pickle
GPT = __import__("10 GPT", fromlist=['GPT']).GPT


# 任务: ① 预测被 mask 掉的词; ② 判断前后句是否存在关联


# 此处 BERT 在判断上下文关系时没有添加特殊符号 "[CLS]", 原因如下:
# 作者认为不用传递给模型一个 "[CLS]" 信息让模型知道当前在做的是什么任务,
# 因为我想要得到的是一个语言理解器,
# 至于对于不同的任务, 可以通过 Finetune 出不同的头来适应,
# 因为谁知道你下游是不是一个你训练过的任务呢？
# 所以我觉得没必要专门为了 Task 去搞一个 Task input,
# 我更关注的是训练出一个语言模型,而不是一个语言任务模型。


class BERT(GPT):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__(model_dim, max_len, n_layer, n_head,
                         n_vocab, lr, max_seg, drop_rate, padding_idx)

        # I think task emb is not necessary for pretraining,
        # because the aim of all tasks is to train a universal sentence embedding
        # the body encoder is the same across all tasks,
        # and different output layer defines different task just like transfer learning.
        # finetuning replaces output layer and leaves the body encoder unchanged.

        # self.task_emb = keras.layers.Embedding(
        #     input_dim=n_task, output_dim=model_dim,  # [n_task, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )

    def step(self, seqs, segs, seqs_, loss_mask, nsp_labels):
        with tf.GradientTape() as tape:
            mlm_logits, nsp_logits = self.call(seqs, segs, training=True)

            # 由于这个模型是对所有位置的词都进行预测,
            # 因此需要用 loss_mask 来告诉模型实际上需要预测的那些词的位置 (也就是被 mask 掉的那些词),
            # 我们只需要取出这些位置预测出来的词做损失就行了
            mlm_loss_batch = tf.boolean_mask(
                self.cross_entropy(seqs_, mlm_logits), loss_mask)
            mlm_loss = tf.reduce_mean(mlm_loss_batch)

            # 作者没有引入分类符号 "[CLS]" 来做 "前后文关系判断" 这个任务,
            # 作者采用的方法是将最后一层输出的所有词的 embedding 同时输入一个全连接层中,
            # 由此来得到分类的概率分布, 进而求取该任务的损失
            nsp_loss = tf.reduce_mean(
                self.cross_entropy(nsp_labels, nsp_logits))

            # 将两个任务的损失合并起来一起训练
            loss = mlm_loss + 0.2 * nsp_loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, mlm_logits

    def mask(self, seqs):
        mask = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # [n, 1, 1, step]


def _get_loss_mask(len_arange, seq, pad_id):
    rand_id = np.random.choice(len_arange, size=max(
        2, int(MASK_RATE * len(len_arange))), replace=False)
    loss_mask = np.full_like(seq, pad_id, dtype=np.bool)
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id


def do_mask(seq, len_arange, pad_id, mask_id):
    # 对输入序列中的词用特殊符号 "[MASK]" 进行替换
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = mask_id
    return loss_mask


def do_replace(seq, len_arange, pad_id, word_ids):
    # 对输入序列中的词用另一个随机的词进行替换
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    seq[rand_id] = np.random.choice(word_ids, size=len(rand_id))
    return loss_mask


def do_nothing(seq, len_arange, pad_id):
    # 不做任何事情
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask


def random_mask_or_replace(data, arange, batch_size):
    seqs, segs, xlen, nsp_labels = data.sample(batch_size)
    seqs_ = seqs.copy()
    p = np.random.random()
    if p < 0.7:  # mask
        loss_mask = np.concatenate(
            [do_mask(
                seqs[i],
                np.concatenate(
                    (arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id,
                data.v2i["<MASK>"]) for i in range(len(seqs))], axis=0)
    elif p < 0.85:  # do nothing
        loss_mask = np.concatenate(
            [do_nothing(
                seqs[i],
                np.concatenate(
                    (arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id) for i in range(len(seqs))], axis=0)
    else:  # replace
        loss_mask = np.concatenate(
            [do_replace(
                seqs[i],
                np.concatenate(
                    (arange[:xlen[i, 0]], arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                data.pad_id,
                data.word_ids) for i in range(len(seqs))], axis=0)
    return seqs, segs, seqs_, loss_mask, xlen, nsp_labels


def train(model, data, step=10000, name="bert"):
    t0 = time.time()
    arange = np.arange(0, data.max_len)
    for t in range(step):
        # 每回合都进行 mask 15% 个词的处理 (或者替换, 或者不做任何事)
        seqs, segs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(
            data, arange, 16)
        loss, pred = model.step(seqs, segs, seqs_, loss_mask, nsp_labels)
        if t % 100 == 0:
            pred = pred[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i]
                                      for i in seqs[0][:xlen[0].sum()+1]]),
                "\n| prd: ", " ".join([data.i2v[i]
                                      for i in pred[:xlen[0].sum()+1]]),
                "\n| tgt word: ", [data.i2v[i] for i in seqs_[
                    0]*loss_mask[0] if i != data.v2i["<PAD>"]],
                "\n| prd word: ", [data.i2v[i] for i in pred *
                                   loss_mask[0] if i != data.v2i["<PAD>"]],
            )
            t0 = t1
    os.makedirs("./visual/models/%s" % name, exist_ok=True)
    model.save_weights("./visual/models/%s/model.ckpt" % name)


def export_attention(model, data, name="bert"):
    model.load_weights("./visual/models/%s/model.ckpt" % name)

    # save attention matrix for visualization
    seqs, segs, xlen, nsp_labels = data.sample(32)
    model.call(seqs, segs, False)
    data = {"src": [[data.i2v[i] for i in seqs[j]]
                    for j in range(len(seqs))], "attentions": model.attentions}
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    utils.set_soft_gpu(True)
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4
    MASK_RATE = 0.15

    d = utils.MRPCData("./MRPC", 2000)
    print("num word: ", d.num_word)
    m = BERT(
        model_dim=MODEL_DIM, max_len=d.max_len, n_layer=N_LAYER, n_head=4, n_vocab=d.num_word,
        lr=LEARNING_RATE, max_seg=d.num_seg, drop_rate=0.2, padding_idx=d.v2i["<PAD>"])
    train(m, d, step=10000, name="bert")
    export_attention(m, d, "bert")
